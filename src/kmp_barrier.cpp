// Copyright 2025 Mu-Tsun Tsai
// SPDX-License-Identifier: MIT

#include "platform.h"

#if NCNN_SIMPLEOMP

#include <stdint.h>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations for OMP runtime functions
extern int omp_get_num_threads();
extern int omp_get_thread_num();

#ifdef __cplusplus
} // extern "C"
#endif

namespace ncnn {

// Barrier state for a team of threads
struct BarrierState
{
    int num_threads;        // Total number of threads in the team
    int arrived;            // Number of threads that have arrived at the barrier
    int generation;         // Barrier generation counter to handle multiple barrier calls
    Mutex lock;
    ConditionVariable condition;
};

// Global map to store barrier states for different teams
// Key is the master thread's address (represents the team)
static std::map<void*, BarrierState*> barrier_states;
static ncnn::Mutex barrier_map_lock;

} // namespace ncnn

#ifdef __cplusplus
extern "C" {
#endif

// Barrier implementation for Clang/LLVM libomp ABI
// All threads in a team must call this function to synchronize
void __kmpc_barrier(void* loc, int32_t gtid)
{
    int num_threads = omp_get_num_threads();

    // Single-threaded case: no barrier needed
    if (num_threads == 1)
    {
        return;
    }

    int thread_num = omp_get_thread_num();

    // Use the location pointer as a unique identifier for this barrier
    // This allows multiple barriers to coexist in different parallel regions
    void* barrier_id = loc;

    ncnn::BarrierState* state = nullptr;

    // Get or create barrier state for this team
    {
        ncnn::barrier_map_lock.lock();

        auto it = ncnn::barrier_states.find(barrier_id);
        if (it == ncnn::barrier_states.end())
        {
            // First thread to arrive creates the barrier state
            state = new ncnn::BarrierState();
            state->num_threads = num_threads;
            state->arrived = 0;
            state->generation = 0;
            ncnn::barrier_states[barrier_id] = state;
        }
        else
        {
            state = it->second;
        }

        ncnn::barrier_map_lock.unlock();
    }

    // Synchronize at the barrier
    {
        state->lock.lock();

        int current_generation = state->generation;
        state->arrived++;

        if (state->arrived == num_threads)
        {
            // Last thread to arrive: reset and wake up all waiting threads
            state->arrived = 0;
            state->generation++;
            state->condition.broadcast();
            state->lock.unlock();
        }
        else
        {
            // Wait for all threads to arrive
            while (current_generation == state->generation)
            {
                state->condition.wait(state->lock);
            }
            state->lock.unlock();
        }
    }

    // Clean up barrier state when all threads have passed
    // This is done by the master thread (thread 0) after synchronization
    if (thread_num == 0)
    {
        ncnn::barrier_map_lock.lock();

        state->lock.lock();
        bool should_cleanup = (state->arrived == 0);
        state->lock.unlock();

        if (should_cleanup)
        {
            ncnn::barrier_states.erase(barrier_id);
            delete state;
        }

        ncnn::barrier_map_lock.unlock();
    }
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEOMP
