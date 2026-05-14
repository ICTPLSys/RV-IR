#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

// Simple timer function
static uint64_t get_time_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// Initialize matrices with simple predictable values for testing
void init_abc(float* A, size_t size_A, float* B, size_t size_B, float* C, size_t size_C, int r_seed_1, int r_seed_2) {
    for (size_t i = 0; i < size_A; i++) {
        A[i] = (float)(i % 10) * 0.1f * r_seed_1;
    }
    for (size_t i = 0; i < size_B; i++) {
        B[i] = (float)((i + 3) % 10) * 0.1f * r_seed_2;
    }
    for (size_t i = 0; i < size_C; i++) {
        C[i] = 123.1f;
    }
}

// Function to run a single GEMM configuration
void run_gemm_benchmark(const char* layer_name, size_t M, size_t N, size_t K) {
    size_t lda = M; // Since A is Col-Major
    size_t ldb = N; // Since B is Row-Major
    size_t ldc = M; // Since C is Col-Major
    
    float alpha = 1.0f;
    float beta = 0.0f; // Beta 0 mode

    // Allocate memory
    size_t size_A = lda * K;
    size_t size_B = ldb * K;
    size_t size_C = ldc * N;
    float *A = (float*)malloc(size_A * sizeof(float));
    float *B = (float*)malloc(size_B * sizeof(float)); 
    float *C = (float*)malloc(size_C * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for %s (%zu x %zu x %zu)\n", layer_name, M, N, K);
        if(A) free(A);
        if(B) free(B);
        if(C) free(C);
        return;
    }

    // Initialize
    init_abc(A, size_A, B, size_B, C, size_C, 1, 1);
    
    // Warmup
    int warmup_iters = 10;
    int measure_iters = 100;

    for (int i = 0; i < warmup_iters; i++) {
        // B is considered Row-Major based on Python offset (k * ldb + n)
        // Since OpenBLAS is configured with CblasColMajor, we must transpose B
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    alpha, A, lda, B, ldb,
                    beta, C, ldc);
    }

    uint64_t start = get_time_ns();
    for (int i = 0; i < measure_iters; i++) {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                    M, N, K,
                    alpha, A, lda, B, ldb,
                    beta, C, ldc);
    }
    uint64_t end = get_time_ns();

    double avg_ns = (double)(end - start) / measure_iters;
    double gflops = (2.0 * M * N * K) / (avg_ns);

    printf("%-20s %8zu %8zu %8zu   | %8.2f us | %8.2f GFLOPS\n", 
           layer_name, M, N, K, avg_ns / 1000.0, gflops);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_dat_file>\n", argv[0]);
        fprintf(stderr, "Example: %s ../cnn_models/bert_large.dat\n", argv[0]);
        return 1;
    }

    const char* filepath = argv[1];
    FILE* file = fopen(filepath, "r");
    if (!file) {
        perror("Failed to open file");
        return 1;
    }

    printf("Benchmarking OpenBLAS SGEMM with file: %s\n", filepath);
    printf("======================================================================================\n");
    printf("%-20s %8s %8s %8s   | %11s | %14s\n", "Layer Name", "M", "N", "K", "Avg Time", "Speed");
    printf("--------------------------------------------------------------------------------------\n");

    char line[512];
    while (fgets(line, sizeof(line), file)) {
        // Skip comments and empty lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }

        char layer_name[256];
        size_t M, N, K;
        
        // Parse layer name and M N K (ignoring flop counts at trailing end if exist)
        if (sscanf(line, " %255s %zu %zu %zu", layer_name, &M, &N, &K) == 4) {
            run_gemm_benchmark(layer_name, M, N, K);
        } else {
            // Check if it's purely M N K without layer name
            if (sscanf(line, " %zu %zu %zu", &M, &N, &K) == 3) {
                 run_gemm_benchmark("Unnamed_Layer", M, N, K);
            }
        }
    }

    fclose(file);
    printf("======================================================================================\n");
    return 0;
}
