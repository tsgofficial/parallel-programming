#include "tasksys.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

// ─── Scale knobs ───────────────────────────────────────────────────────────
static constexpr int ARRAY_SIZE = 1 << 24; // 16 M elements  (was 1 M)
static constexpr int NUM_SUBTASKS = 1024;  // finer grain    (was 256)
static constexpr int REPEAT = 20;          // repeated runs  (was 1)
// ──────────────────────────────────────────────────────────────────────────

static void runBenchmark(const char *label, ITaskSystem &sys)
{
    std::vector<float> arr(ARRAY_SIZE, 0.0f);

    auto work = [&arr](int task_id, int num_total_tasks)
    {
        int chunk = ARRAY_SIZE / num_total_tasks;
        int lo = task_id * chunk;
        int hi = (task_id == num_total_tasks - 1) ? ARRAY_SIZE : lo + chunk;
        for (int i = lo; i < hi; ++i)
            arr[i] = std::sin(static_cast<float>(i)) * 0.5f + 0.25f;
    };

    // Warm-up (not timed)
    sys.run(work, NUM_SUBTASKS);

    using Clock = std::chrono::steady_clock;

    // ── REPEAT runs: exposes thread-creation overhead per call ──────────
    double total_elapsed = 0.0;
    double best_elapsed = 1e9;

    for (int r = 0; r < REPEAT; ++r)
    {
        auto t0 = Clock::now();
        sys.run(work, NUM_SUBTASKS);
        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        total_elapsed += elapsed;
        if (elapsed < best_elapsed)
            best_elapsed = elapsed;
    }

    double avg_ms = total_elapsed / REPEAT * 1000.0;
    double best_ms = best_elapsed * 1000.0;

    std::cout << std::left << std::setw(30) << label
              << "  avg=" << std::fixed << std::setprecision(2)
              << std::setw(8) << avg_ms << " ms"
              << "  best=" << std::setw(8) << best_ms << " ms"
              << "  (" << REPEAT << " runs)\n";
}

// ─── Spawn overhead stress test ───────────────────────────────────────────
// Calls run() many times with a TINY workload — this is where Spawn loses
// badly because it pays thread-creation cost every single call.
static void spawnOverheadTest(const char *label, ITaskSystem &sys)
{
    static constexpr int SMALL_N = 1 << 14; // 16 K elements
    static constexpr int SMALL_TASKS = 64;
    static constexpr int MANY_CALLS = 200;

    std::vector<float> arr(SMALL_N, 0.0f);
    auto work = [&arr](int task_id, int num_total_tasks)
    {
        int chunk = SMALL_N / num_total_tasks;
        int lo = task_id * chunk;
        int hi = (task_id == num_total_tasks - 1) ? SMALL_N : lo + chunk;
        for (int i = lo; i < hi; ++i)
            arr[i] = std::sin(static_cast<float>(i));
    };

    // Warm-up
    for (int i = 0; i < 3; ++i)
        sys.run(work, SMALL_TASKS);

    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    for (int c = 0; c < MANY_CALLS; ++c)
        sys.run(work, SMALL_TASKS);
    double total_ms =
        std::chrono::duration<double>(Clock::now() - t0).count() * 1000.0;

    std::cout << std::left << std::setw(30) << label
              << "  total=" << std::fixed << std::setprecision(2)
              << std::setw(8) << total_ms << " ms"
              << "  per_call=" << std::setw(7) << total_ms / MANY_CALLS << " ms"
              << "  (" << MANY_CALLS << " calls x " << SMALL_TASKS << " tasks)\n";
}

// ─── Part II test: dependency chain A → {B,C} → D ─────────────────────────
static void testDeps(ITaskSystem &sys)
{
    std::atomic<int> counter{0};

    auto makeWork = [&counter](int mult) -> IRunnable
    {
        return [&counter, mult](int, int)
        { counter.fetch_add(mult); };
    };

    TaskID A = sys.runAsyncWithDeps(makeWork(1), 128, {});
    TaskID B = sys.runAsyncWithDeps(makeWork(2), 2, {A});
    TaskID C = sys.runAsyncWithDeps(makeWork(3), 6, {A});
    TaskID D = sys.runAsyncWithDeps(makeWork(4), 32, {B, C});
    sys.sync();

    int expected = 128 * 1 + 2 * 2 + 6 * 3 + 32 * 4; // 278
    std::cout << "  Deps test: counter=" << counter.load()
              << " expected=" << expected
              << (counter.load() == expected ? "  [PASS]" : "  [FAIL]")
              << "\n";
    (void)D;
}

// ─── Chained deps stress: long serial chain of async tasks ─────────────────
static void testDepsChain(ITaskSystem &sys)
{
    static constexpr int CHAIN_LEN = 16;
    std::atomic<int> counter{0};

    auto work = [&counter](int, int)
    { counter.fetch_add(1); };

    TaskID prev = sys.runAsyncWithDeps(work, 10, {});
    for (int i = 1; i < CHAIN_LEN; ++i)
        prev = sys.runAsyncWithDeps(work, 10, {prev});
    sys.sync();

    int expected = CHAIN_LEN * 10;
    std::cout << "  Chain test (" << CHAIN_LEN << " steps x 10 tasks): "
              << "counter=" << counter.load()
              << " expected=" << expected
              << (counter.load() == expected ? "  [PASS]" : "  [FAIL]")
              << "\n";
}

int main()
{
    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    const int NT = std::max(2, hw);

    std::cout << "Hardware threads: " << hw
              << "  |  Using: " << NT << " threads\n";
    std::cout << "Array size: " << (ARRAY_SIZE >> 20)
              << " M floats  |  Subtasks: " << NUM_SUBTASKS
              << "  |  Repeats: " << REPEAT << "\n\n";

    // ── Benchmark 1: Large array, repeated runs ─────────────────────────
    std::cout << "=== Benchmark 1: Large workload (" << REPEAT << " repeats) ===\n";
    {
        TaskSystemSerial sys(1);
        runBenchmark("Serial", sys);
    }
    {
        TaskSystemParallelSpawn sys(NT);
        runBenchmark("Parallel Spawn", sys);
    }
    {
        TaskSystemParallelThreadPoolSpinning sys(NT);
        runBenchmark("Spinning Pool", sys);
    }
    {
        TaskSystemParallelThreadPoolSleeping sys(NT);
        runBenchmark("Sleeping Pool", sys);
    }

    // ── Part II: Async with dependencies ────────────────────────────────
    std::cout << "\n=== Part II: Async with Dependencies ===\n";
    {
        TaskSystemParallelAsyncWithDeps sys(NT);
        testDeps(sys);
        testDepsChain(sys);
    }

    return 0;
}