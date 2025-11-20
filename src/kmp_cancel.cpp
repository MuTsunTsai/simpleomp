// Copyright 2025 Mu-Tsun Tsai
// SPDX-License-Identifier: MIT

// OpenMP Cancellation Support
// Implements __kmpc_cancel and __kmpc_cancellationpoint for LLVM libomp ABI

#include "platform.h"

#if NCNN_SIMPLEOMP

#include <map>
#include <atomic>
#include <stdio.h>

#define DEBUG_CANCEL 0

// Forward declaration
extern "C" int omp_get_cancellation(void);

#ifdef __cplusplus
extern "C" {
#endif

// Cancellation kind constants (from OpenMP specification)
enum {
    cancel_noreq = 0,     // No cancellation request
    cancel_parallel = 1,  // Cancel parallel region
    cancel_loop = 2,      // Cancel loop (for/do)
    cancel_sections = 3,  // Cancel sections
    cancel_taskgroup = 4  // Cancel taskgroup
};

// Global cancellation state
// SimpleOMP doesn't support nested parallelism, so we only need one global flag
// for the currently active parallel region
static std::atomic<int> g_current_cancel_kind(cancel_noreq);

// NOTE: This simple approach works because:
// 1. SimpleOMP doesn't support nested parallel regions
// 2. Parallel regions execute sequentially (one finishes before the next starts)
// 3. Therefore, only one team can be active at any time

/*!
@ingroup CANCELLATION
@param loc source location information
@param gtid global thread id
@param cncl_kind the kind of cancellation (parallel, for, sections, taskgroup)

@return returns non-zero if the encountering thread has to cancel, zero otherwise

The cancel construct activates cancellation of the binding region.
If cancellation is activated, the encountering task will begin executing the
cancellation sequence and the encountering thread will ultimately resume execution
at the end of the canceled region.
*/
int __kmpc_cancel(void* loc, int gtid, int cncl_kind)
{
    (void)loc; // Unused in SimpleOMP

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL] Thread %d calling __kmpc_cancel, kind=%d\n", gtid, cncl_kind);
    #endif

    // Check if cancellation is enabled (controlled by OMP_CANCELLATION env var)
    if (!omp_get_cancellation()) {
        #if DEBUG_CANCEL
        fprintf(stderr, "[CANCEL] Thread %d: cancellation disabled\n", gtid);
        #endif
        return 0; // Cancellation is disabled, ignore
    }

    // Get current cancellation state
    int current_kind = g_current_cancel_kind.load(std::memory_order_acquire);

    // If no cancellation is active, or if we're setting a stronger cancellation
    // (cancel_parallel cancels everything, so it takes precedence)
    if (current_kind == cancel_noreq || cncl_kind == cancel_parallel) {
        g_current_cancel_kind.store(cncl_kind, std::memory_order_release);
    }

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL] Thread %d: set cancel_kind=%d, returning 1\n", gtid, cncl_kind);
    #endif

    // Return non-zero to indicate this thread should cancel
    return 1;
}

/*!
@ingroup CANCELLATION
@param loc source location information
@param gtid global thread id
@param cncl_kind the kind of cancellation point (parallel, for, sections, taskgroup)

@return returns non-zero if a matching cancellation request has been flagged

A cancellation point allows a thread to check if cancellation has been requested
for the innermost enclosing region of the specified type.
*/
int __kmpc_cancellationpoint(void* loc, int gtid, int cncl_kind)
{
    (void)loc; // Unused in SimpleOMP

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCELLATION_POINT] Thread %d checking, kind=%d\n", gtid, cncl_kind);
    #endif

    // Check if cancellation is enabled
    if (!omp_get_cancellation()) {
        return 0; // Cancellation is disabled
    }

    // Get current cancellation state
    int current_kind = g_current_cancel_kind.load(std::memory_order_acquire);

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCELLATION_POINT] Thread %d: current_cancel_kind=%d\n", gtid, current_kind);
    #endif

    // Check if cancellation matches the requested kind
    // cancel_parallel cancels everything
    if (current_kind == cancel_parallel) {
        #if DEBUG_CANCEL
        fprintf(stderr, "[CANCELLATION_POINT] Thread %d: returning 1 (cancel_parallel)\n", gtid);
        #endif
        return 1;
    }

    // Otherwise, only cancel if the kind matches exactly
    if (current_kind == cncl_kind) {
        #if DEBUG_CANCEL
        fprintf(stderr, "[CANCELLATION_POINT] Thread %d: returning 1 (exact match)\n", gtid);
        #endif
        return 1;
    }

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCELLATION_POINT] Thread %d: returning 0 (no match)\n", gtid);
    #endif
    return 0;
}

/*!
@ingroup CANCELLATION
Clear the cancellation flag for the current team (called when region ends)
*/
void __kmpc_cancel_clear(void* loc)
{
    (void)loc; // Unused in SimpleOMP

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL_CLEAR] Clearing cancellation state\n");
    #endif

    // Reset the global cancellation flag
    g_current_cancel_kind.store(cancel_noreq, std::memory_order_release);
}

/*!
@ingroup CANCELLATION
@param loc source location information
@param gtid global thread id

Barrier with cancellation point
This is a special barrier that also checks for cancellation
*/
int __kmpc_cancel_barrier(void* loc, int gtid)
{
    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL_BARRIER] Thread %d entering\n", gtid);
    #endif

    // First check for cancellation
    int cancelled = __kmpc_cancellationpoint(loc, gtid, cancel_parallel);

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL_BARRIER] Thread %d: cancellation check returned %d\n", gtid, cancelled);
    #endif

    // If cancelled, skip the barrier and return immediately
    if (cancelled) {
        #if DEBUG_CANCEL
        fprintf(stderr, "[CANCEL_BARRIER] Thread %d: skipping barrier due to cancellation\n", gtid);
        #endif
        return cancelled;
    }

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL_BARRIER] Thread %d: calling normal barrier\n", gtid);
    #endif

    // Otherwise, perform normal barrier
    extern void __kmpc_barrier(void* loc, int gtid);
    __kmpc_barrier(loc, gtid);

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL_BARRIER] Thread %d: barrier completed, checking cancellation again\n", gtid);
    #endif

    // Check again after barrier in case another thread cancelled
    int result = __kmpc_cancellationpoint(loc, gtid, cancel_parallel);

    #if DEBUG_CANCEL
    fprintf(stderr, "[CANCEL_BARRIER] Thread %d: exiting, returning %d\n", gtid, result);
    #endif

    return result;
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // NCNN_SIMPLEOMP
