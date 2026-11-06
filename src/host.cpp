/**
* Copyright (C) 2019-2021 Xilinx, Inc
* SPDX-License-Identifier: MIT
*
* Licensed under the Apache License, Version 2.0 (the "License"). You may
* not use this file except in compliance with the License. A copy of the
* License is located at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*/

#include "cmdlineparser.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

// XRT includes

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

const int NUM_ENGINES = 8;  // 총 PE 쌍 개수(dispatcher-comp)
const int CHUNKS_PER_ENGINE = 32;                          // 데이터 배치 단위(blake3 의 pass). SHA-256 커널에는 pass 가 없다
const int CHUNKS_PER_PASS = NUM_ENGINES * CHUNKS_PER_ENGINE; // 1 pass = 256 청크

// ─── SW 골든: SHA-256 (FIPS 180-4) — hls/test.cpp 의 sw_sha256 과 같은 코드 ─────────
static const uint32_t SW_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

static inline uint32_t sw_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void sw_sha256(const uint8_t* msg, uint64_t len, uint8_t digest[32]) {
    uint32_t H[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                     0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    // 패딩: 0x80, 0 들, 길이(비트, big-endian 8바이트) → 길이가 64 의 배수가 되게
    uint64_t padded_len = ((len + 8) / 64 + 1) * 64;
    std::vector<uint8_t> buf(padded_len, 0);
    if (len) memcpy(buf.data(), msg, len);
    buf[len] = 0x80;
    for (int i = 0; i < 8; i++) buf[padded_len - 1 - i] = (uint8_t)((len * 8) >> (8 * i));

    for (uint64_t off = 0; off < padded_len; off += 64) {
        uint32_t W[64];
        for (int t = 0; t < 16; t++) {
            const uint8_t* p = &buf[off + 4 * t];
            W[t] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
        }
        for (int t = 16; t < 64; t++) {
            uint32_t s0 = sw_rotr(W[t - 15], 7) ^ sw_rotr(W[t - 15], 18) ^ (W[t - 15] >> 3);
            uint32_t s1 = sw_rotr(W[t - 2], 17) ^ sw_rotr(W[t - 2], 19) ^ (W[t - 2] >> 10);
            W[t] = W[t - 16] + s0 + W[t - 7] + s1;
        }

        uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
        for (int t = 0; t < 64; t++) {
            uint32_t S1  = sw_rotr(e, 6) ^ sw_rotr(e, 11) ^ sw_rotr(e, 25);
            uint32_t ch  = (e & f) ^ (~e & g);
            uint32_t T1  = h + S1 + ch + SW_K[t] + W[t];
            uint32_t S0  = sw_rotr(a, 2) ^ sw_rotr(a, 13) ^ sw_rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t T2  = S0 + maj;
            h = g; g = f; f = e; e = d + T1;
            d = c; c = b; b = a; a = T1 + T2;
        }
        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    for (int i = 0; i < 8; i++) {
        digest[4 * i + 0] = (uint8_t)(H[i] >> 24);
        digest[4 * i + 1] = (uint8_t)(H[i] >> 16);
        digest[4 * i + 2] = (uint8_t)(H[i] >> 8);
        digest[4 * i + 3] = (uint8_t)(H[i]);
    }
}

int main(int argc, char** argv) {
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    // Switches
    //**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.addSwitch("--num_chunks", "-n", "number of chunks to process (multiple of 256, and of NUM_ENGINES*S)", std::to_string(2 * CHUNKS_PER_PASS));
    parser.addSwitch("--chunks_per_segment", "-s", "chunks per segment (power of two, 4 <= S <= 1024)", "4");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));
    uint64_t num_chunks = stoull(parser.value("num_chunks"));
    uint32_t chunks_per_segment = (uint32_t)stoul(parser.value("chunks_per_segment"));

    // num_chunks는 CHUNKS_PER_PASS(=NUM_ENGINES*32=256)의 배수.
    //   SHA-256 커널에 필요한 제약은 아니지만, blake3 와 같은 (N, S) 점에서 비교하려고 같은 제약을 둔다.
    if (num_chunks % CHUNKS_PER_PASS != 0) {
        std::cerr << "Error: num_chunks must be a multiple of " << CHUNKS_PER_PASS << "." << std::endl;
        return EXIT_FAILURE;
    }

    // chunks_per_segment는 2의 거듭제곱이면서 4 <= S <= 1024
    if ((chunks_per_segment & (chunks_per_segment - 1)) != 0 ||
        chunks_per_segment < 4 || chunks_per_segment > 1024) {
        std::cerr << "Error: chunks_per_segment must be a power of two with 4 <= S <= 1024." << std::endl;
        return EXIT_FAILURE;
    }

    // block 배치 제약: 엔진 e의 담당 구간 base = e*(N/NUM_ENGINES)가 S의 배수여야
    //   세그먼트가 엔진 경계를 넘지 않는다 (엔진마다 세그먼트 N/(8S)개를 통째로 해시한다).
    if (num_chunks % ((uint64_t)NUM_ENGINES * chunks_per_segment) != 0) {
        std::cerr << "Error: num_chunks must be a multiple of NUM_ENGINES*chunks_per_segment ("
                  << (uint64_t)NUM_ENGINES * chunks_per_segment << ")." << std::endl;
        return EXIT_FAILURE;
    }

    if (argc < 3) {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    uint64_t num_passes = num_chunks / CHUNKS_PER_PASS;
    uint64_t blocks_per_pe = num_passes * 512; // 한 PE당 한 패스에 512개 블락 처리(32개 청크)
    uint64_t total_blocks = num_chunks * 16;
    uint64_t num_segments = num_chunks / chunks_per_segment; // 출력 해시 개수

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "sha256_accelerator");

    std::cout << "Allocate Buffer in Global Memory\n";
    size_t in_size_bytes = sizeof(uint32_t) * 16 * blocks_per_pe; // 4B * 16 = one block size
    // 이것도 uint64_t로 캐스팅 됨
    size_t out_size_bytes = sizeof(uint32_t) * 8 * num_segments;   // 4B * 8 = one hash

    // 커널 인자 인덱스: host_data_in_0..7 = 0..7, host_hash_out = 8
    auto in_bo_0 = xrt::bo(device, in_size_bytes, krnl.group_id(0));
    auto in_bo_1 = xrt::bo(device, in_size_bytes, krnl.group_id(1));
    auto in_bo_2 = xrt::bo(device, in_size_bytes, krnl.group_id(2));
    auto in_bo_3 = xrt::bo(device, in_size_bytes, krnl.group_id(3));
    auto in_bo_4 = xrt::bo(device, in_size_bytes, krnl.group_id(4));
    auto in_bo_5 = xrt::bo(device, in_size_bytes, krnl.group_id(5));
    auto in_bo_6 = xrt::bo(device, in_size_bytes, krnl.group_id(6));
    auto in_bo_7 = xrt::bo(device, in_size_bytes, krnl.group_id(7));
    auto out_bo  = xrt::bo(device, out_size_bytes, krnl.group_id(8));

    // Map the contents of the buffer object into host memory
    uint32_t* in_map_0 = in_bo_0.map<uint32_t*>();
    uint32_t* in_map_1 = in_bo_1.map<uint32_t*>();
    uint32_t* in_map_2 = in_bo_2.map<uint32_t*>();
    uint32_t* in_map_3 = in_bo_3.map<uint32_t*>();
    uint32_t* in_map_4 = in_bo_4.map<uint32_t*>();
    uint32_t* in_map_5 = in_bo_5.map<uint32_t*>();
    uint32_t* in_map_6 = in_bo_6.map<uint32_t*>();
    uint32_t* in_map_7 = in_bo_7.map<uint32_t*>();
    uint32_t* out_map  = out_bo.map<uint32_t*>();

    std::vector<uint32_t> sw_in(total_blocks * 16); //한 블락이 16*4B(64B). 그걸 total_blocks만큼 만들어라
    std::vector<uint8_t> sw_out(num_segments * 32); // 세그먼트마다 해시 32B

    printf("[Host] Generating Data...\n");

    // [v3 block] global_chunk_idx = e*(N/NUM_ENGINES) + p*32 + c
    //   in_map_0 = chunk 0 ~ N/8-1 통째로, in_map_1 = chunk N/8 ~ N/4-1 통째로, ...
    //   엔진 e 는 자기 포트를 처음부터 순서대로 읽어 세그먼트 e*N/(8S) ~ (e+1)*N/(8S)-1 을 해시한다.
    for (uint64_t p = 0; p < num_passes; p++) {
        for (int e = 0; e < NUM_ENGINES; e++) {
            for (int c = 0; c < 32; c++) {
                uint64_t global_chunk_idx = (uint64_t)e * (num_chunks / NUM_ENGINES) + p * 32 + c;

                for (int b = 0; b < 16; b++) {
                    uint64_t local_idx = p * 512 + c * 16 + b;

                    for (int w = 0; w < 16; w++) {
                        uint64_t val = (global_chunk_idx * 16 + b + w) * 0x11223344;
                        sw_in[(global_chunk_idx * 16 + b) * 16 + w] = val; // 1차원 평탄화 인덱싱

                        if (e == 0) in_map_0[local_idx * 16 + w] = val;
                        if (e == 1) in_map_1[local_idx * 16 + w] = val;
                        if (e == 2) in_map_2[local_idx * 16 + w] = val;
                        if (e == 3) in_map_3[local_idx * 16 + w] = val;
                        if (e == 4) in_map_4[local_idx * 16 + w] = val;
                        if (e == 5) in_map_5[local_idx * 16 + w] = val;
                        if (e == 6) in_map_6[local_idx * 16 + w] = val;
                        if (e == 7) in_map_7[local_idx * 16 + w] = val;
                    }
                }
            }
        }
    }

    for (uint64_t m = 0; m < num_segments; m++) {
    // 세그먼트 m = 청크 m*S ~ (m+1)*S-1. x86 은 little-endian 이라 sw_in 의 메모리 바이트가 HBM 에 들어가는 바이트와 같다
    sw_sha256((const uint8_t*)&sw_in[m * chunks_per_segment * 256], (uint64_t)chunks_per_segment * 1024, &sw_out[m * 32]);
    }

    // Synchronize buffer content with device side
    std::cout << "\n[Host -> Device] Transferring data to HBM...\n";
    auto h2d_start = std::chrono::steady_clock::now();
    in_bo_0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_3.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_4.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_5.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_6.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    in_bo_7.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto h2d_end = std::chrono::steady_clock::now();

    std::cout << "[Kernel] Running SHA-256 Accelerator...\n";
    auto krnl_start = std::chrono::steady_clock::now();
    auto run = krnl(in_bo_0, in_bo_1, in_bo_2, in_bo_3,
                    in_bo_4, in_bo_5, in_bo_6, in_bo_7,
                    out_bo, num_chunks, chunks_per_segment);
    run.wait();
    auto krnl_end = std::chrono::steady_clock::now();

    std::cout << "[Device -> Host] Retrieving Hash Output...\n\n";
    auto d2h_start = std::chrono::steady_clock::now();
    out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto d2h_end = std::chrono::steady_clock::now();

    std::chrono::duration<double> h2d_time = h2d_end - h2d_start;
    std::chrono::duration<double> krnl_time = krnl_end - krnl_start;
    std::chrono::duration<double> d2h_time = d2h_end - d2h_start;
    std::cout << "H2D Transfer Time: " << h2d_time.count() << " s"<<std::endl;
    std::cout << "Kernel Exec: " << krnl_time.count() << " s" <<std::endl;
    std::cout << "D2H Transfer Time: " << d2h_time.count() << " s" <<std::endl;

    std::cout<<std::endl;

    std::cout << "[Host] Running Software Golden Reference...\n";

    // 세그먼트마다 32B 해시 1개. word로 쪼개지 않고 세그먼트당 한 줄로 비교한다
    int errors = 0;
    std::cout << "\n[Result Comparison] num_chunks: " << num_chunks
               << ", chunks_per_segment: " << chunks_per_segment
               << ", num_segments: " << num_segments << "\n";
    for (uint64_t m = 0; m < num_segments; m++) {
        const uint8_t* hw = (const uint8_t*)&out_map[m * 8]; // 해시 32B (sha256sum 과 같은 바이트 순서)
        int mismatch = memcmp(hw, &sw_out[m * 32], 32) != 0;

        printf("[seg %5llu] chunk %6llu~%6llu  HW=",
               (unsigned long long)m,
               (unsigned long long)(m * chunks_per_segment),
               (unsigned long long)((m + 1) * chunks_per_segment - 1));
        for (int i = 0; i < 32; i++) printf("%02x", hw[i]);
        printf("  %s\n", mismatch ? "<<< MISMATCH" : "MATCH");

        if (mismatch) {
            printf("                          SW=");
            for (int i = 0; i < 32; i++) printf("%02x", sw_out[m * 32 + i]);
            printf("\n");
            errors++;
        }
    }

    std::cout<<std::endl;

    if (errors == 0) {
        std::cout << ">>> TEST PASSED! HW Output perfectly matches SW Golden Reference.\n";
    } else {
        std::cout << ">>> TEST FAILED! Found " << errors << " mismatches.\n";
    }

    double port_data_gb = (double)in_size_bytes / 1e9; // GB 단위 변환 (10^9)
    double total_data_gb = port_data_gb * NUM_ENGINES; // NUM_ENGINES개의 입력 포트 사용

    double effective_bw_per_port = port_data_gb / krnl_time.count(); // GB/s
    double total_effective_bw = total_data_gb / krnl_time.count();   // GB/s

    std::cout << "\n================ [ Performance Report ] ================\n";
    std::cout << "Data Size per Port : " << port_data_gb * 1000 << " MB\n";
    std::cout << "Total Data Size    : " << total_data_gb * 1000 << " MB\n";
    std::cout << "Effective BW (Port): " << effective_bw_per_port << " GB/s\n";
    std::cout << "Effective BW (Total): " << total_effective_bw << " GB/s\n";
    std::cout << "========================================================\n\n";

    return errors;
}
