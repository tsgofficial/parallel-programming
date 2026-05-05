#pragma once
#include <functional>
#include <vector>

// Unique ID for each async task (Part II)
using TaskID = int;

// The unit of work: a callable that receives its sub-task index
using IRunnable = std::function<void(int, int)>;  // (task_id, num_total_tasks)

/*
 * ITaskSystem – Abstract base class for all task systems.
 *
 * Part I  : run()  – synchronous bulk launch
 * Part II : runAsyncWithDeps() / sync() – async with dependency tracking
 */
class ITaskSystem {
public:
    explicit ITaskSystem(int num_threads) : num_threads_(num_threads) {}
    virtual ~ITaskSystem() = default;

    // Part I – run num_total_tasks instances of runnable, block until done
    virtual void run(IRunnable runnable, int num_total_tasks) = 0;

    // Part II – queue a bulk task, return its ID; do NOT block
    virtual TaskID runAsyncWithDeps(IRunnable           runnable,
                                    int                  num_total_tasks,
                                    const std::vector<TaskID>& deps) = 0;

    // Part II – block until every previously submitted task has finished
    virtual void sync() = 0;

protected:
    int num_threads_;
};
