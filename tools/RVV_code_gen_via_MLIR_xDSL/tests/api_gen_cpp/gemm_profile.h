#ifndef GEMM_PROFILE_H
#define GEMM_PROFILE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// ============================================================
// Timer Functions - Using clock_gettime for compatibility
// ============================================================

// Use POSIX clock_gettime() - works on most RISC-V systems
static inline uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#define TIMER_READ() get_time_ns()
#define TIMER_UNIT "ns"

// ============================================================
// Profiling Data Structures
// ============================================================

typedef struct {
    double *samples;       // Array of timing samples (double for sub-ns precision)
    int num_samples;       // Number of samples collected
    int capacity;          // Array capacity
    uint64_t total_flops;  // Total FLOPs for this operation
} profile_data_t;

// Initialize profiling data
static inline profile_data_t* profile_init(int capacity, uint64_t flops) {
    profile_data_t *p = (profile_data_t*)malloc(sizeof(profile_data_t));
    p->samples = (double*)malloc(capacity * sizeof(double));
    p->num_samples = 0;
    p->capacity = capacity;
    p->total_flops = flops;
    return p;
}

// Add a timing sample
static inline void profile_add_sample(profile_data_t *p, double time) {
    if (p->num_samples < p->capacity) {
        p->samples[p->num_samples++] = time;
    }
}

// Comparison function for qsort
static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

// Compute statistics
static inline void profile_compute_stats(profile_data_t *p,
                                         double *min, double *median,
                                         double *mean, double *stddev) {
    if (p->num_samples == 0) return;
    
    // Sort samples for median
    qsort(p->samples, p->num_samples, sizeof(double), compare_double);
    
    // Min
    *min = p->samples[0];
    
    // Median
    *median = p->samples[p->num_samples / 2];
    
    // Mean
    double sum = 0;
    for (int i = 0; i < p->num_samples; i++) {
        sum += p->samples[i];
    }
    *mean = sum / p->num_samples;
    
    // Standard deviation
    double variance = 0.0;
    for (int i = 0; i < p->num_samples; i++) {
        double diff = p->samples[i] - (*mean);
        variance += diff * diff;
    }
    *stddev = sqrt(variance / p->num_samples);
}

// Print performance report
static inline void profile_print_report(const char *kernel_name, 
                                        profile_data_t *p,
                                        double cpu_freq_ghz) {
    double min, median, mean;
    double stddev;
    
    profile_compute_stats(p, &min, &median, &mean, &stddev);
    
    // Convert to microseconds
    double time_min_us = min / 1000.0;
    double time_median_us = median / 1000.0;
    
    // FLOP/s calculation
    double time_sec = min / 1e9;
    double gflops_min = (double)p->total_flops / time_sec / 1e9;
    
    time_sec = median / 1e9;
    double gflops_median = (double)p->total_flops / time_sec / 1e9;
    
    // Compact one-line format: Kernel | Min(µs) | Median(µs) | Peak(GFLOP/s) | Median(GFLOP/s)
printf("%-15s %8.2f µs %8.2f µs  %8.3f GFLOP/s %8.3f GFLOP/s\n",
           kernel_name, time_min_us, time_median_us, gflops_min, gflops_median);
}

// Free profiling data
static inline void profile_free(profile_data_t *p) {
    free(p->samples);
    free(p);
}

// Print comparison of xDSL vs Reference
static inline void profile_print_comparison(const char *kernel_base_name,
                                           profile_data_t *xdsl_prof,
                                           profile_data_t *ref_prof) {
    double xdsl_min, xdsl_median, xdsl_mean;
    double ref_min, ref_median, ref_mean;
    double xdsl_stddev, ref_stddev;
    
    profile_compute_stats(xdsl_prof, &xdsl_min, &xdsl_median, &xdsl_mean, &xdsl_stddev);
    profile_compute_stats(ref_prof, &ref_min, &ref_median, &ref_mean, &ref_stddev);
    
    // Convert to microseconds
    double xdsl_min_us = xdsl_min / 1000.0;
    double ref_min_us = ref_min / 1000.0;
    
    // FLOP/s calculation
    double xdsl_gflops = (double)xdsl_prof->total_flops / (xdsl_min / 1e9) / 1e9;
    double ref_gflops = (double)ref_prof->total_flops / (ref_min / 1e9) / 1e9;
    
    // Speedup
    double speedup = ref_min_us / xdsl_min_us;
    
    // Single line: Kernel | xDSL time | Ref time | | xDSL GF/s | Ref GF/s | Speedup
    printf("%-15s %8.6f %8.6f   %8.6f %8.6f %7.4fx\n",
           kernel_base_name, xdsl_min_us, ref_min_us, xdsl_gflops, ref_gflops, speedup);
}

// ============================================================
// Profiling Macros
// ============================================================

#define PROFILE_KERNEL(kernel_name, kernel_call, mr, nr, kc, warmup, iterations, cpu_freq) \
do { \
    uint64_t flops = (uint64_t)(mr) * (nr) * (kc) * 2; \
    profile_data_t *prof = profile_init(iterations, flops); \
    \
    /* Warmup */ \
    for (int _w = 0; _w < (warmup); _w++) { \
        kernel_call; \
    } \
    \
    /* Measurement */ \
    for (int _i = 0; _i < (iterations); _i++) { \
        uint64_t _start = TIMER_READ(); \
        kernel_call; \
        uint64_t _end = TIMER_READ(); \
        profile_add_sample(prof, _end - _start); \
    } \
    \
    profile_print_report(kernel_name, prof, cpu_freq); \
    profile_free(prof); \
} while(0)

#endif // GEMM_PROFILE_H
