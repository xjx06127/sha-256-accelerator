#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "sha224_256_fixed.hpp" // Vitis Security Library L1 사본. 원본과 다른 곳은 dup_strm, sha256Digest loop_flatten off 두 곳 (MODIFICATIONS.md)

// ─── HW 아키텍처 상수 (blake3_accelerator 와 같은 값) ────────────────────
const int NUM_ENGINES      = 8;  // 총 엔진 수 (dispatcher-comp 쌍) = HBM 입력 포트 수
const int BLOCKS_PER_CHUNK = 16; // 청크(1KiB) 당 블록(64B) 수
const int WORDS_PER_BLOCK  = 8;  // 블록(512b) 하나 = 라이브러리 입력 워드(64b) 8개

// ─── FIFO depth 상수 ──────────────────────────────
// 코어가 블록당 69 cycle(cosim 실측)로 느려서 깊을 필요가 없다.
const int FIFO_DEPTH_MSG  = 16; // disp_to_comp_msg_0..7     : 64b 워드, 블록 2개분
const int FIFO_DEPTH_LEN  = 4;  // disp_to_comp_len/end_0..7 : 세그먼트마다 1개
const int FIFO_DEPTH_HASH = 4;  // comp_to_wr(_end)_0..7       : 세그먼트마다 해시 1개

// ─── cosim 용 m_axi depth (하드웨어에는 영향 없음) ───────────────────────
// hls/test.cpp 의 TB_MAX_NUM_CHUNKS 와 반드시 같게. 테스트벤치 버퍼도 이 크기로 잡는다.
const int TB_MAX_NUM_CHUNKS = 8192;
const int AXI_DEPTH_IN  = TB_MAX_NUM_CHUNKS / NUM_ENGINES * BLOCKS_PER_CHUNK; // 엔진당 블록 수 = 16384
const int AXI_DEPTH_OUT = TB_MAX_NUM_CHUNKS / 4;                              // 해시 수 최대 (S=4) = 2048

typedef hls::vector<uint32_t, 16> block_vec_t; // 512b = 블록 1개 (blake3 와 같음)
typedef hls::vector<uint32_t, 8>  cv_vec_t;    // 256b = 해시 1개 (blake3 와 같음)
typedef ap_uint<64>  msg_word_t;               // 라이브러리 입력 워드. m_width=64 가 원본이 지원하는 최대
typedef ap_uint<256> hash_t;                   // 라이브러리 출력 해시

// 엔진당 세그먼트 수 = N / (8·S).
// S 가 2의 거듭제곱(4..1024)이라 나눗셈 대신 shift 로 구한다 (DSP divider 회피, blake3 cv_pe 와 같은 방식).
static uint64_t segs_per_engine(uint64_t num_chunks, uint32_t chunks_per_segment) {
    #pragma HLS INLINE
    uint32_t seg_log2 = 0;
    for (int b = 0; b < 11; b++) {
        #pragma HLS UNROLL
        if (chunks_per_segment & (1u << b)) seg_log2 = b;
    }
    return (num_chunks >> seg_log2) / NUM_ENGINES;
}

// ══════════════════════════════════════════════════════════════════════════
//  dispatcher_pe : 엔진 e 의 HBM 버퍼(청크 [e·N/8, (e+1)·N/8), 호스트가 배치)를 처음부터 끝까지 읽어
//                  라이브러리 입력 형식으로 넘긴다.
//    - 512b 블록 1개를 64b 워드 8개로 나눈다 (라이브러리 입력이 64b 까지라서). 바이트 순서는 그대로.
//    - 세그먼트 첫 블록에서 그 세그먼트의 길이(S·1KiB)와 end=false 를 보낸다. 다 끝나면 end=true.
//    - 블록 하나에 8 cycle (II=8). 코어가 블록당 69 cycle 이라 병목이 아니다.
// ══════════════════════════════════════════════════════════════════════════
void dispatcher_pe(const block_vec_t* ext_mem,
                   hls::stream<msg_word_t>& msg_fifo,
                   hls::stream<ap_uint<64> >& len_fifo,
                   hls::stream<bool>& end_len_fifo,
                   uint64_t num_chunks,
                   uint32_t chunks_per_segment) {
    const uint64_t num_blocks     = num_chunks / NUM_ENGINES * BLOCKS_PER_CHUNK;       // 이 엔진의 블록 수
    const uint64_t blocks_per_seg = (uint64_t)chunks_per_segment * BLOCKS_PER_CHUNK;  // 2의 거듭제곱
    const uint64_t seg_bytes      = blocks_per_seg * 64;

    for (uint64_t i = 0; i < num_blocks; i++) {
        #pragma HLS PIPELINE II=8
        if ((i & (blocks_per_seg - 1)) == 0) { // 세그먼트 첫 블록
            len_fifo.write(seg_bytes);
            end_len_fifo.write(false);
        }

        block_vec_t blk = ext_mem[i];
        for (int j = 0; j < WORDS_PER_BLOCK; j++) {
            #pragma HLS UNROLL
            msg_word_t w;
            w.range(31, 0)  = blk[2 * j];     // 앞 4바이트
            w.range(63, 32) = blk[2 * j + 1]; // 뒤 4바이트
            msg_fifo.write(w);
        }
    }
    end_len_fifo.write(true);
}

// comp_pe : SHA-256 코어. 라이브러리를 부르기만 한다 (사본: dup_strm 수정 + digest 블록 루프 loop_flatten off, 나머지는 원본 그대로).
void comp_pe(hls::stream<msg_word_t>& msg_fifo,
             hls::stream<ap_uint<64> >& len_fifo,
             hls::stream<bool>& end_len_fifo,
             hls::stream<hash_t>& hash_fifo,
             hls::stream<bool>& end_hash_fifo) {
    xf::security::sha256<64>(msg_fifo, len_fifo, end_len_fifo, hash_fifo, end_hash_fifo);
}

// ══════════════════════════════════════════════════════════════════════════
//  hash_writer_pe : 엔진 8개의 해시를 엔진 순서대로 한 개씩(round-robin) 읽어 host_hash_out[세그먼트 번호] 에 쓴다.
//    모든 엔진이 같은 수의 세그먼트를 같은 속도로 처리하므로, k 번째 바퀴에 엔진 e 가 내는 해시 = 그 엔진의 k 번째 세그먼트
//    = 세그먼트 e·segs + k (호스트의 블록 배치). gmem8 은 이 프로세스만 쓴다.
//    라이브러리가 해시마다 붙이는 end 표시(해시마다 false, 마지막에 true)도 여기서 소비한다.
//    blake3 는 512b 쌍이 고속으로 몰려 relay 트리를 두었지만, 여기는 해시가 세그먼트당 1개뿐이라 바로 읽는다.
// ══════════════════════════════════════════════════════════════════════════
void hash_writer_pe(hls::stream<hash_t> hash_fifo[NUM_ENGINES],
                    hls::stream<bool> end_fifo[NUM_ENGINES],
                    cv_vec_t* ext_out,
                    uint64_t num_chunks,
                    uint32_t chunks_per_segment) {
    const uint64_t segs = segs_per_engine(num_chunks, chunks_per_segment);

    uint64_t k    = 0; // 바퀴 번호
    uint64_t base = 0; // e·segs
    int      e    = 0; // 엔진 번호

    for (uint64_t i = 0; i < NUM_ENGINES * segs; i++) {
        #pragma HLS PIPELINE II=1
        end_fifo[e].read(); // false
        hash_t h = hash_fifo[e].read();

        cv_vec_t out; // 해시 byte i 가 bits [8i+7:8i] → 32b 워드로 그대로 옮기면 메모리에는 표준 순서
        for (int w = 0; w < 8; w++) {
            #pragma HLS UNROLL
            out[w] = h.range(32 * w + 31, 32 * w);
        }
        ext_out[base + k] = out;

        if (e == NUM_ENGINES - 1) { e = 0; base = 0; k++; }
        else                      { e++;   base += segs; }
    }
    // 마지막 end=true
    for (int j = 0; j < NUM_ENGINES; j++) {
        #pragma HLS UNROLL
        end_fifo[j].read();
    }
}

void sha256_accelerator(const block_vec_t* host_data_in_0,
                        const block_vec_t* host_data_in_1,
                        const block_vec_t* host_data_in_2,
                        const block_vec_t* host_data_in_3,
                        const block_vec_t* host_data_in_4,
                        const block_vec_t* host_data_in_5,
                        const block_vec_t* host_data_in_6,
                        const block_vec_t* host_data_in_7,
                        cv_vec_t* host_hash_out,
                        uint64_t  num_chunks,
                        uint32_t  chunks_per_segment) {
    #pragma HLS INTERFACE m_axi port=host_data_in_0 depth=AXI_DEPTH_IN  bundle=gmem0
    #pragma HLS INTERFACE m_axi port=host_data_in_1 depth=AXI_DEPTH_IN  bundle=gmem1
    #pragma HLS INTERFACE m_axi port=host_data_in_2 depth=AXI_DEPTH_IN  bundle=gmem2
    #pragma HLS INTERFACE m_axi port=host_data_in_3 depth=AXI_DEPTH_IN  bundle=gmem3
    #pragma HLS INTERFACE m_axi port=host_data_in_4 depth=AXI_DEPTH_IN  bundle=gmem4
    #pragma HLS INTERFACE m_axi port=host_data_in_5 depth=AXI_DEPTH_IN  bundle=gmem5
    #pragma HLS INTERFACE m_axi port=host_data_in_6 depth=AXI_DEPTH_IN  bundle=gmem6
    #pragma HLS INTERFACE m_axi port=host_data_in_7 depth=AXI_DEPTH_IN  bundle=gmem7
    #pragma HLS INTERFACE m_axi port=host_hash_out  depth=AXI_DEPTH_OUT bundle=gmem8

    // dispatcher_pe -> comp_pe : 라이브러리 입력 (메시지 워드, 길이, 끝 표시)
    hls::stream<msg_word_t>   disp_to_comp_msg[NUM_ENGINES];
    hls::stream<ap_uint<64> > disp_to_comp_len[NUM_ENGINES];
    hls::stream<bool>         disp_to_comp_end[NUM_ENGINES];
    #pragma HLS STREAM variable=disp_to_comp_msg depth=FIFO_DEPTH_MSG
    #pragma HLS STREAM variable=disp_to_comp_len depth=FIFO_DEPTH_LEN
    #pragma HLS STREAM variable=disp_to_comp_end depth=FIFO_DEPTH_LEN

    // comp_pe -> hash_writer_pe : 라이브러리 출력 (해시, 끝 표시)
    hls::stream<hash_t> comp_to_wr[NUM_ENGINES];
    hls::stream<bool>   comp_to_wr_end[NUM_ENGINES];
    #pragma HLS STREAM variable=comp_to_wr     depth=FIFO_DEPTH_HASH
    #pragma HLS STREAM variable=comp_to_wr_end depth=FIFO_DEPTH_HASH

    #pragma HLS DATAFLOW
    dispatcher_pe(host_data_in_0, disp_to_comp_msg[0], disp_to_comp_len[0], disp_to_comp_end[0], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_1, disp_to_comp_msg[1], disp_to_comp_len[1], disp_to_comp_end[1], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_2, disp_to_comp_msg[2], disp_to_comp_len[2], disp_to_comp_end[2], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_3, disp_to_comp_msg[3], disp_to_comp_len[3], disp_to_comp_end[3], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_4, disp_to_comp_msg[4], disp_to_comp_len[4], disp_to_comp_end[4], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_5, disp_to_comp_msg[5], disp_to_comp_len[5], disp_to_comp_end[5], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_6, disp_to_comp_msg[6], disp_to_comp_len[6], disp_to_comp_end[6], num_chunks, chunks_per_segment);
    dispatcher_pe(host_data_in_7, disp_to_comp_msg[7], disp_to_comp_len[7], disp_to_comp_end[7], num_chunks, chunks_per_segment);

    comp_pe(disp_to_comp_msg[0], disp_to_comp_len[0], disp_to_comp_end[0], comp_to_wr[0], comp_to_wr_end[0]);
    comp_pe(disp_to_comp_msg[1], disp_to_comp_len[1], disp_to_comp_end[1], comp_to_wr[1], comp_to_wr_end[1]);
    comp_pe(disp_to_comp_msg[2], disp_to_comp_len[2], disp_to_comp_end[2], comp_to_wr[2], comp_to_wr_end[2]);
    comp_pe(disp_to_comp_msg[3], disp_to_comp_len[3], disp_to_comp_end[3], comp_to_wr[3], comp_to_wr_end[3]);
    comp_pe(disp_to_comp_msg[4], disp_to_comp_len[4], disp_to_comp_end[4], comp_to_wr[4], comp_to_wr_end[4]);
    comp_pe(disp_to_comp_msg[5], disp_to_comp_len[5], disp_to_comp_end[5], comp_to_wr[5], comp_to_wr_end[5]);
    comp_pe(disp_to_comp_msg[6], disp_to_comp_len[6], disp_to_comp_end[6], comp_to_wr[6], comp_to_wr_end[6]);
    comp_pe(disp_to_comp_msg[7], disp_to_comp_len[7], disp_to_comp_end[7], comp_to_wr[7], comp_to_wr_end[7]);

    hash_writer_pe(comp_to_wr, comp_to_wr_end, host_hash_out, num_chunks, chunks_per_segment);
}
