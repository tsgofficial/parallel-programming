#pragma once
#include "itasksys.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
// 0.  Serial  (reference implementation, already correct)
// ─────────────────────────────────────────────────────────────────────────────
class TaskSystemSerial : public ITaskSystem {
public:
    explicit TaskSystemSerial(int num_threads);
    ~TaskSystemSerial() override = default;

    void   run(IRunnable runnable, int num_total_tasks) override;
    TaskID runAsyncWithDeps(IRunnable runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps) override;
    void   sync() override;
};

// ─────────────────────────────────────────────────────────────────────────────
// 1.  Parallel Spawn  – spawn fresh threads on every run() call
// ─────────────────────────────────────────────────────────────────────────────
class TaskSystemParallelSpawn : public ITaskSystem {
public:
    explicit TaskSystemParallelSpawn(int num_threads);
    ~TaskSystemParallelSpawn() override = default;

    void   run(IRunnable runnable, int num_total_tasks) override;
    TaskID runAsyncWithDeps(IRunnable runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps) override;
    void   sync() override;
};

// ─────────────────────────────────────────────────────────────────────────────
// 2.  Parallel Thread-Pool (Spinning)
//     Workers busy-loop waiting for tasks; protected with a mutex.
// ─────────────────────────────────────────────────────────────────────────────
class TaskSystemParallelThreadPoolSpinning : public ITaskSystem {
public:
    explicit TaskSystemParallelThreadPoolSpinning(int num_threads);
    ~TaskSystemParallelThreadPoolSpinning() override;

    void   run(IRunnable runnable, int num_total_tasks) override;
    TaskID runAsyncWithDeps(IRunnable runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps) override;
    void   sync() override;

private:
    // Shared state visible to all workers
    std::vector<std::thread> workers_;

    // Current batch
    IRunnable   current_runnable_;
    int         num_total_tasks_{0};
    std::atomic<int> next_task_{0};       // next sub-task index to grab
    std::atomic<int> tasks_done_{0};      // how many sub-tasks finished
    std::atomic<bool> batch_ready_{false}; // is a batch waiting?
    std::atomic<bool> stop_{false};

    // Protects next_task_ fetch-and-increment
    std::mutex task_mutex_;

    void workerThread();
};

// ─────────────────────────────────────────────────────────────────────────────
// 3.  Parallel Thread-Pool (Sleeping)
//     Workers sleep on a condition variable when idle.
// ─────────────────────────────────────────────────────────────────────────────
class TaskSystemParallelThreadPoolSleeping : public ITaskSystem {
public:
    explicit TaskSystemParallelThreadPoolSleeping(int num_threads);
    ~TaskSystemParallelThreadPoolSleeping() override;

    void   run(IRunnable runnable, int num_total_tasks) override;
    TaskID runAsyncWithDeps(IRunnable runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps) override;
    void   sync() override;

private:
    std::vector<std::thread> workers_;

    // Task queue for individual sub-tasks (index into current batch)
    std::queue<int>  task_queue_;
    IRunnable        current_runnable_;
    int              num_total_tasks_{0};
    std::atomic<int> tasks_done_{0};

    std::mutex              queue_mutex_;
    std::condition_variable worker_cv_;   // wakes workers when tasks arrive
    std::condition_variable master_cv_;   // wakes master when batch finishes
    bool                    stop_{false};

    void workerThread();
};

// ─────────────────────────────────────────────────────────────────────────────
// 4.  Async With Dependencies  (Part II)
//     Uses the Sleeping pool internally; adds dependency tracking.
// ─────────────────────────────────────────────────────────────────────────────
struct AsyncTask {
    TaskID              id;
    IRunnable           runnable;
    int                 num_total_tasks;
    std::vector<TaskID> deps;           // IDs we're waiting for
    std::atomic<int>    sub_done{0};    // completed sub-tasks
    bool                finished{false};
};

class TaskSystemParallelAsyncWithDeps : public ITaskSystem {
public:
    explicit TaskSystemParallelAsyncWithDeps(int num_threads);
    ~TaskSystemParallelAsyncWithDeps() override;

    void   run(IRunnable runnable, int num_total_tasks) override;
    TaskID runAsyncWithDeps(IRunnable runnable, int num_total_tasks,
                            const std::vector<TaskID>& deps) override;
    void   sync() override;

private:
    std::vector<std::thread> workers_;

    // All registered tasks
    std::unordered_map<TaskID, AsyncTask*> all_tasks_;
    // Tasks whose deps are all met – ready to execute
    std::queue<std::pair<AsyncTask*, int>> ready_queue_; // (task, sub_index)
    // Tasks still waiting for deps
    std::vector<AsyncTask*> waiting_;

    std::atomic<TaskID> next_id_{0};
    std::atomic<int>    tasks_in_flight_{0}; // sub-tasks not yet finished

    std::mutex              mtx_;
    std::condition_variable worker_cv_;
    std::condition_variable sync_cv_;
    bool                    stop_{false};

    void workerThread();
    void tryPromoteWaiting();  // move tasks whose deps are met into ready_queue_
};
