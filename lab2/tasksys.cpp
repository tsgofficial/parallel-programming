#include "tasksys.h"

#include <algorithm>
#include <cassert>

// ═════════════════════════════════════════════════════════════════════════════
// 0. TaskSystemSerial
// ═════════════════════════════════════════════════════════════════════════════
TaskSystemSerial::TaskSystemSerial(int num_threads)
    : ITaskSystem(num_threads) {}

void TaskSystemSerial::run(IRunnable runnable, int num_total_tasks)
{
    for (int i = 0; i < num_total_tasks; ++i)
        runnable(i, num_total_tasks);
}

TaskID TaskSystemSerial::runAsyncWithDeps(IRunnable runnable,
                                          int num_total_tasks,
                                          const std::vector<TaskID> &)
{
    run(std::move(runnable), num_total_tasks);
    return 0;
}

void TaskSystemSerial::sync() {}

// ═════════════════════════════════════════════════════════════════════════════
// 1. TaskSystemParallelSpawn
//    Weakness: thread creation/destruction overhead on every run() call.
// ═════════════════════════════════════════════════════════════════════════════
TaskSystemParallelSpawn::TaskSystemParallelSpawn(int num_threads)
    : ITaskSystem(num_threads) {}

void TaskSystemParallelSpawn::run(IRunnable runnable, int num_total_tasks)
{
    int nthreads = std::min(num_threads_, num_total_tasks);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    int chunk = (num_total_tasks + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t)
    {
        int lo = t * chunk;
        int hi = std::min(lo + chunk, num_total_tasks);
        if (lo >= num_total_tasks)
            break;
        threads.emplace_back([&runnable, lo, hi, num_total_tasks]()
                             {
            for (int i = lo; i < hi; ++i)
                runnable(i, num_total_tasks); });
    }
    for (auto &th : threads)
        th.join();
}

TaskID TaskSystemParallelSpawn::runAsyncWithDeps(IRunnable runnable,
                                                 int num_total_tasks,
                                                 const std::vector<TaskID> &)
{
    run(std::move(runnable), num_total_tasks);
    return 0;
}

void TaskSystemParallelSpawn::sync() {}

// ═════════════════════════════════════════════════════════════════════════════
// 2. TaskSystemParallelThreadPoolSpinning
//
// FIX: Removed the redundant task_mutex_ around next_task_ fetch_add.
//      fetch_add is already atomic — the mutex was causing unnecessary
//      serialization, making all workers queue up for a lock on every
//      single sub-task. Now fully lock-free in the hot path.
//
// FIX: Worker loop restructured so the idx check and execute are one
//      clean atomic grab — no load() + separate fetch_add() race window.
// ═════════════════════════════════════════════════════════════════════════════
TaskSystemParallelThreadPoolSpinning::TaskSystemParallelThreadPoolSpinning(
    int num_threads)
    : ITaskSystem(num_threads)
{
    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i)
        workers_.emplace_back(
            &TaskSystemParallelThreadPoolSpinning::workerThread, this);
}

TaskSystemParallelThreadPoolSpinning::~TaskSystemParallelThreadPoolSpinning()
{
    stop_.store(true);
    for (auto &th : workers_)
        th.join();
}

void TaskSystemParallelThreadPoolSpinning::workerThread()
{
    while (!stop_.load(std::memory_order_relaxed))
    {
        if (!batch_ready_.load(std::memory_order_acquire))
            continue;

        // Atomically claim the next sub-task index — no mutex needed.
        int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
        if (idx < num_total_tasks_)
        {
            current_runnable_(idx, num_total_tasks_);
            tasks_done_.fetch_add(1, std::memory_order_release);
        }
        // If idx >= num_total_tasks_ this worker just spins back to the top,
        // waiting for either a new batch or stop_.
    }
}

void TaskSystemParallelThreadPoolSpinning::run(IRunnable runnable,
                                               int num_total_tasks)
{
    current_runnable_ = std::move(runnable);
    num_total_tasks_ = num_total_tasks;
    next_task_.store(0, std::memory_order_relaxed);
    tasks_done_.store(0, std::memory_order_relaxed);

    // Release workers.
    batch_ready_.store(true, std::memory_order_release);

    // Master thread also participates — same lock-free grab.
    while (true)
    {
        int idx = next_task_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= num_total_tasks_)
            break;
        current_runnable_(idx, num_total_tasks_);
        tasks_done_.fetch_add(1, std::memory_order_release);
    }

    // Spin until all workers have finished their claimed tasks.
    while (tasks_done_.load(std::memory_order_acquire) < num_total_tasks_)
        ; // intentional busy-wait — this is the Spinning variant

    batch_ready_.store(false, std::memory_order_release);
}

TaskID TaskSystemParallelThreadPoolSpinning::runAsyncWithDeps(
    IRunnable runnable, int num_total_tasks,
    const std::vector<TaskID> &)
{
    run(std::move(runnable), num_total_tasks);
    return 0;
}

void TaskSystemParallelThreadPoolSpinning::sync() {}

// ═════════════════════════════════════════════════════════════════════════════
// 3. TaskSystemParallelThreadPoolSleeping
//    Workers sleep on condition_variable; master is woken when batch finishes.
// ═════════════════════════════════════════════════════════════════════════════
TaskSystemParallelThreadPoolSleeping::TaskSystemParallelThreadPoolSleeping(
    int num_threads)
    : ITaskSystem(num_threads)
{
    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i)
        workers_.emplace_back(
            &TaskSystemParallelThreadPoolSleeping::workerThread, this);
}

TaskSystemParallelThreadPoolSleeping::~TaskSystemParallelThreadPoolSleeping()
{
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        stop_ = true;
    }
    worker_cv_.notify_all();
    for (auto &th : workers_)
        th.join();
}

void TaskSystemParallelThreadPoolSleeping::workerThread()
{
    while (true)
    {
        int task_idx = -1;
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            worker_cv_.wait(lk, [this]
                            { return stop_ || !task_queue_.empty(); });
            if (stop_ && task_queue_.empty())
                return;
            task_idx = task_queue_.front();
            task_queue_.pop();
        }

        current_runnable_(task_idx, num_total_tasks_);

        int done = tasks_done_.fetch_add(1) + 1;
        if (done == num_total_tasks_)
        {
            master_cv_.notify_one();
        }
    }
}

void TaskSystemParallelThreadPoolSleeping::run(IRunnable runnable,
                                               int num_total_tasks)
{
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        current_runnable_ = std::move(runnable);
        num_total_tasks_ = num_total_tasks;
        tasks_done_.store(0);
        for (int i = 0; i < num_total_tasks; ++i)
            task_queue_.push(i);
    }
    worker_cv_.notify_all();

    std::unique_lock<std::mutex> lk(queue_mutex_);
    master_cv_.wait(lk, [this]
                    { return tasks_done_.load() == num_total_tasks_; });
}

TaskID TaskSystemParallelThreadPoolSleeping::runAsyncWithDeps(
    IRunnable runnable, int num_total_tasks,
    const std::vector<TaskID> &)
{
    run(std::move(runnable), num_total_tasks);
    return 0;
}

void TaskSystemParallelThreadPoolSleeping::sync() {}

// ═════════════════════════════════════════════════════════════════════════════
// 4. TaskSystemParallelAsyncWithDeps  (Part II)
// ═════════════════════════════════════════════════════════════════════════════
TaskSystemParallelAsyncWithDeps::TaskSystemParallelAsyncWithDeps(int num_threads)
    : ITaskSystem(num_threads)
{
    workers_.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i)
        workers_.emplace_back(
            &TaskSystemParallelAsyncWithDeps::workerThread, this);
}

TaskSystemParallelAsyncWithDeps::~TaskSystemParallelAsyncWithDeps()
{
    {
        std::lock_guard<std::mutex> lk(mtx_);
        stop_ = true;
    }
    worker_cv_.notify_all();
    for (auto &th : workers_)
        th.join();

    for (auto &[id, t] : all_tasks_)
        delete t;
}

// Move tasks from waiting_ whose deps are all finished into ready_queue_.
// Must be called with mtx_ held.
void TaskSystemParallelAsyncWithDeps::tryPromoteWaiting()
{
    bool promoted = true;
    while (promoted)
    {
        promoted = false;
        for (auto it = waiting_.begin(); it != waiting_.end();)
        {
            AsyncTask *t = *it;
            bool ready = std::all_of(t->deps.begin(), t->deps.end(),
                                     [this](TaskID dep_id)
                                     {
                                         auto found = all_tasks_.find(dep_id);
                                         return found != all_tasks_.end() && found->second->finished;
                                     });
            if (ready)
            {
                for (int i = 0; i < t->num_total_tasks; ++i)
                {
                    ready_queue_.push({t, i});
                    tasks_in_flight_.fetch_add(1);
                }
                it = waiting_.erase(it);
                promoted = true;
            }
            else
            {
                ++it;
            }
        }
    }
}

void TaskSystemParallelAsyncWithDeps::workerThread()
{
    while (true)
    {
        AsyncTask *task = nullptr;
        int sub_i = -1;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            worker_cv_.wait(lk, [this]
                            { return stop_ || !ready_queue_.empty(); });
            if (stop_ && ready_queue_.empty())
                return;
            auto [t, idx] = ready_queue_.front();
            ready_queue_.pop();
            task = t;
            sub_i = idx;
        }

        task->runnable(sub_i, task->num_total_tasks);

        int done = task->sub_done.fetch_add(1) + 1;
        if (done == task->num_total_tasks)
        {
            {
                std::lock_guard<std::mutex> lk(mtx_);
                task->finished = true;
                tryPromoteWaiting();
            }
            worker_cv_.notify_all();
        }

        int in_flight = tasks_in_flight_.fetch_sub(1) - 1;
        if (in_flight == 0)
        {
            sync_cv_.notify_all();
        }
    }
}

void TaskSystemParallelAsyncWithDeps::run(IRunnable runnable,
                                          int num_total_tasks)
{
    TaskID id = runAsyncWithDeps(std::move(runnable), num_total_tasks, {});
    sync();
    (void)id;
}

TaskID TaskSystemParallelAsyncWithDeps::runAsyncWithDeps(
    IRunnable runnable, int num_total_tasks,
    const std::vector<TaskID> &deps)
{

    TaskID id = next_id_.fetch_add(1);
    auto *t = new AsyncTask{id, std::move(runnable), num_total_tasks, deps};

    {
        std::lock_guard<std::mutex> lk(mtx_);
        all_tasks_[id] = t;

        bool ready = std::all_of(deps.begin(), deps.end(),
                                 [this](TaskID dep_id)
                                 {
                                     auto found = all_tasks_.find(dep_id);
                                     return found != all_tasks_.end() && found->second->finished;
                                 });

        if (ready)
        {
            for (int i = 0; i < num_total_tasks; ++i)
            {
                ready_queue_.push({t, i});
                tasks_in_flight_.fetch_add(1);
            }
        }
        else
        {
            waiting_.push_back(t);
        }
    }
    worker_cv_.notify_all();
    return id;
}

void TaskSystemParallelAsyncWithDeps::sync()
{
    std::unique_lock<std::mutex> lk(mtx_);
    sync_cv_.wait(lk, [this]
                  { return tasks_in_flight_.load() == 0 && waiting_.empty(); });
}