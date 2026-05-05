/**
 * Laboratory Work #1 - Parallel Computing
 * Task A: Sum of 80M elements (serial, OpenMP, std::thread)
 * Task B: Element-wise transform sin(A[i])*0.5+0.25 (serial, OpenMP, std::thread)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <cmath>
#include <thread>
#include <algorithm>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

// ─── Timing Utilities ────────────────────────────────────────────────────────
// Using steady_clock to ensure measurements aren't affected by system time jumps.
using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

// Calculates the time passed from t0 until "now" in milliseconds.
static double elapsed_ms(Clock::time_point t0)
{
    return Ms(Clock::now() - t0).count();
}

// ─── Constants ───────────────────────────────────────────────────────────────
// 80 million elements (~610MB) ensures we exceed CPU L3 cache limits.
constexpr std::size_t N = 80'000'000;
constexpr int WARMUP = 3; // Runs to "heat up" the CPU caches/clocks.
constexpr int RUNS = 10;  // Number of timed iterations for averaging.

// ─── Statistical Helpers ─────────────────────────────────────────────────────
// Returns the 95th percentile: 95% of runs were faster than this value.
static double p95(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    std::size_t idx = static_cast<std::size_t>(0.95 * (v.size() - 1));
    return v[idx];
}

static double mean(const std::vector<double> &v)
{
    double s = 0;
    for (auto x : v)
        s += x;
    return s / v.size();
}

// ─── Task A: Summation Implementations ───────────────────────────────────────

// Standard single-threaded loop.
double sum_serial(const std::vector<double> &A)
{
    double s = 0.0;
    for (auto x : A)
        s += x;
    return s;
}

// OpenMP version: Uses a compiler directive to split the loop.
// 'reduction(+:s)' prevents race conditions by giving each thread a local sum
// and then adding them all together at the end.
double sum_openmp(const std::vector<double> &A, int nthreads)
{
    double s = 0.0;
#pragma omp parallel for reduction(+ : s) num_threads(nthreads)
    for (std::size_t i = 0; i < A.size(); ++i)
        s += A[i];
    return s;
}

// Manual std::thread version: We manually calculate index ranges (chunks).
double sum_threads(const std::vector<double> &A, int nthreads)
{
    std::size_t n = A.size();
    // Round up division to ensure all elements are covered.
    std::size_t chunk = (n + nthreads - 1) / nthreads;

    // partial[t] stores the result for thread t to avoid "False Sharing"
    // and race conditions on a single global sum variable.
    std::vector<double> partial(nthreads, 0.0);
    std::vector<std::thread> ts;
    ts.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t)
    {
        std::size_t lo = t * chunk;
        std::size_t hi = std::min(lo + chunk, n); // Ensure we don't go out of bounds.

        // Spawn thread with a lambda capturing local bounds and the partial vector.
        ts.emplace_back([&, t, lo, hi]()
                        {
            double s = 0.0;
            for (std::size_t i = lo; i < hi; ++i) s += A[i];
            partial[t] = s; });
    }
    // Wait for all threads to finish.
    for (auto &th : ts)
        th.join();

    // Final reduction: sum the results from each thread.
    double total = 0.0;
    for (auto v : partial)
        total += v;
    return total;
}

// ─── Task B: Element-wise Transform ──────────────────────────────────────────
// This is "compute-bound" due to the sin() function calls.

void transform_serial(std::vector<double> &A)
{
    for (auto &x : A)
        x = std::sin(x) * 0.5 + 0.25;
}

// Simple parallel for loop. No reduction needed because each index is independent.
void transform_openmp(std::vector<double> &A, int nthreads)
{
#pragma omp parallel for num_threads(nthreads)
    for (std::size_t i = 0; i < A.size(); ++i)
        A[i] = std::sin(A[i]) * 0.5 + 0.25;
}

// Manual threading: similar chunking logic to the Sum task.
void transform_threads(std::vector<double> &A, int nthreads)
{
    std::size_t n = A.size();
    std::size_t chunk = (n + nthreads - 1) / nthreads;
    std::vector<std::thread> ts;
    ts.reserve(nthreads);

    for (int t = 0; t < nthreads; ++t)
    {
        std::size_t lo = t * chunk;
        std::size_t hi = std::min(lo + chunk, n);
        ts.emplace_back([&A, lo, hi]()
                        {
            for (std::size_t i = lo; i < hi; ++i)
                A[i] = std::sin(A[i]) * 0.5 + 0.25; });
    }
    for (auto &th : ts)
        th.join();
}

// ─── Logging and Main Logic ──────────────────────────────────────────────────

// Data structure to hold benchmark results for CSV exporting.
struct Record
{
    std::string workload, impl;
    int threads;
    std::size_t size;
    int run_i;
    double elapsed_ms;
    std::string notes;
};

// Helper to dump all collected records into a spreadsheet-ready format.
static void write_csv(const std::string &path, const std::vector<Record> &recs)
{
    std::ofstream f(path);
    f << "workload,impl,threads,size,run_i,elapsed_ms,notes\n";
    for (auto &r : recs)
    {
        f << r.workload << ',' << r.impl << ',' << r.threads << ','
          << r.size << ',' << r.run_i << ','
          << std::fixed << std::setprecision(3) << r.elapsed_ms << ',' << r.notes << '\n';
    }
}

int main()
{
    // Detect how many CPU cores/logical processors the machine has.
    const int max_hw = static_cast<int>(std::thread::hardware_concurrency());

    // Generate a list of thread counts to test (e.g., 1, 2, 4, 8...).
    std::vector<int> thread_counts;
    for (int t = 1; t <= max_hw; t *= 2)
        thread_counts.push_back(t);
    if (thread_counts.back() != max_hw)
        thread_counts.push_back(max_hw);

    std::cout << "Hardware threads: " << max_hw << "\n";
    std::cout << "Array size: " << N << " doubles (" << N * sizeof(double) / 1024 / 1024 << " MB)\n\n";

    // Allocate memory. A_orig is the "source", A is the "destination/workspace".
    std::vector<double> A_orig(N, 1.0);
    std::vector<double> A(N);
    std::vector<Record> records;

    // ─────────────────────────────────────────────────────────────────────────
    // TASK A – SUMMATION BENCHMARK
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "=== Task A: Sum ===\n";

    // --- Serial Run ---
    {
        // Use 'volatile' to prevent the compiler from optimizing away the loop.
        for (int w = 0; w < WARMUP; ++w)
            volatile double r = sum_serial(A_orig);

        std::vector<double> times;
        double last_res = 0;
        for (int i = 0; i < RUNS; ++i)
        {
            auto t0 = Clock::now();
            double r = sum_serial(A_orig);
            double ms = elapsed_ms(t0);
            last_res = r;
            times.push_back(ms);
            records.push_back({"sum", "serial", 1, N, i, ms, "res=" + std::to_string(r)});
        }
        printf("  serial       mean=%.2f ms  p95=%.2f ms  result=%.1f\n", mean(times), p95(times), last_res);
    }

    // --- Parallel Runs (OpenMP & Threads) ---
    for (int nt : thread_counts)
    {
        // Benchmark OpenMP Sum
        {
            for (int w = 0; w < WARMUP; ++w)
                volatile double r = sum_openmp(A_orig, nt);
            std::vector<double> times;
            double last_res = 0;
            for (int i = 0; i < RUNS; ++i)
            {
                auto t0 = Clock::now();
                double r = sum_openmp(A_orig, nt);
                double ms = elapsed_ms(t0);
                last_res = r;
                times.push_back(ms);
                records.push_back({"sum", "openmp", nt, N, i, ms, "res=" + std::to_string(r)});
            }
            printf("  openmp  t=%2d  mean=%.2f ms  p95=%.2f ms  result=%.1f\n", nt, mean(times), p95(times), last_res);
        }
        // Benchmark std::thread Sum
        {
            for (int w = 0; w < WARMUP; ++w)
                volatile double r = sum_threads(A_orig, nt);
            std::vector<double> times;
            double last_res = 0;
            for (int i = 0; i < RUNS; ++i)
            {
                auto t0 = Clock::now();
                double r = sum_threads(A_orig, nt);
                double ms = elapsed_ms(t0);
                last_res = r;
                times.push_back(ms);
                records.push_back({"sum", "threads", nt, N, i, ms, "res=" + std::to_string(r)});
            }
            printf("  threads t=%2d  mean=%.2f ms  p95=%.2f ms  result=%.1f\n", nt, mean(times), p95(times), last_res);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TASK B – TRANSFORMATION BENCHMARK
    // ─────────────────────────────────────────────────────────────────────────
    std::cout << "\n=== Task B: Element-wise Transform ===\n";

    // Helper to reset the array so every run starts with the same data.
    auto reset = [&]()
    { std::fill(A.begin(), A.end(), 1.0); };

    // --- Serial Run ---
    {
        std::vector<double> times;
        double checksum = 0;
        for (int w = 0; w < WARMUP; ++w)
        {
            reset();
            transform_serial(A);
        }
        for (int i = 0; i < RUNS; ++i)
        {
            reset();
            auto t0 = Clock::now();
            transform_serial(A);
            double ms = elapsed_ms(t0);
            checksum = A[0]; // Check first element to verify math.
            times.push_back(ms);
            records.push_back({"transform", "serial", 1, N, i, ms, "A[0]=" + std::to_string(checksum)});
        }
        printf("  serial       mean=%.2f ms  p95=%.2f ms  A[0]=%.6f\n", mean(times), p95(times), checksum);
    }

    // --- Parallel Runs (OpenMP & Threads) ---
    for (int nt : thread_counts)
    {
        // Benchmark OpenMP Transform
        {
            for (int w = 0; w < WARMUP; ++w)
            {
                reset();
                transform_openmp(A, nt);
            }
            std::vector<double> times;
            double checksum = 0;
            for (int i = 0; i < RUNS; ++i)
            {
                reset();
                auto t0 = Clock::now();
                transform_openmp(A, nt);
                double ms = elapsed_ms(t0);
                checksum = A[0];
                times.push_back(ms);
                records.push_back({"transform", "openmp", nt, N, i, ms, "A[0]=" + std::to_string(checksum)});
            }
            printf("  openmp  t=%2d  mean=%.2f ms  p95=%.2f ms  A[0]=%.6f\n", nt, mean(times), p95(times), checksum);
        }
        // Benchmark std::thread Transform
        {
            for (int w = 0; w < WARMUP; ++w)
            {
                reset();
                transform_threads(A, nt);
            }
            std::vector<double> times;
            double checksum = 0;
            for (int i = 0; i < RUNS; ++i)
            {
                reset();
                auto t0 = Clock::now();
                transform_threads(A, nt);
                double ms = elapsed_ms(t0);
                checksum = A[0];
                times.push_back(ms);
                records.push_back({"transform", "threads", nt, N, i, ms, "A[0]=" + std::to_string(checksum)});
            }
            printf("  threads t=%2d  mean=%.2f ms  p95=%.2f ms  A[0]=%.6f\n", nt, mean(times), p95(times), checksum);
        }
    }

    // Write all data points to results.csv for visualization.
    const std::string csv_path = "results.csv";
    write_csv(csv_path, records);
    std::cout << "\nResults saved to " << csv_path << "\n";

    return 0;
}