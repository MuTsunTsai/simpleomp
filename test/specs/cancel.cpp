// Copyright 2025 Mu-Tsun Tsai
// Licensed under the MIT License

#include <gtest/gtest.h>
#include <omp.h>
#include <atomic>
#include <vector>
#include <cmath>

class CancelTest : public testing::Test {
protected:
    void SetUp() override {
        // Cancellation tests require cancellation to be enabled
        // This is controlled by OMP_CANCELLATION environment variable
        if (!omp_get_cancellation()) {
            GTEST_SKIP() << "Cancellation is disabled. Set OMP_CANCELLATION=true to run these tests.";
        }

        // Ensure we have multiple threads for meaningful tests
        ASSERT_GE(omp_get_max_threads(), 2) << "Need at least 2 threads to test cancellation";

        // Note: Cancellation state is now automatically cleared by __kmpc_fork_call
        // at the end of each parallel region (no manual cleanup needed)
    }
};

// Test 1: Basic cancel parallel - one thread cancels, others should detect
// Expected: Threads detect cancellation at cancellation points and exit early
TEST_F(CancelTest, BasicCancelParallel) {
    std::atomic<int> thread_started{0};
    std::atomic<int> thread_completed{0};

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        thread_started++;

        // Thread 0 immediately requests cancellation and exits
        if (tid == 0) {
            #pragma omp cancel parallel
            // Should not reach here - cancel includes implicit cancellation point
            thread_completed++;
        } else {
            // Other threads check for cancellation
            // They will detect the flag set by Thread 0
            #pragma omp cancellation point parallel

            // Threads that reach here were not cancelled (possible due to timing)
            thread_completed++;
        }
    }

    // At least one thread should have started (thread 0)
    EXPECT_GE(thread_started.load(), 1);

    // Thread 0 should have exited immediately without incrementing completed
    // Other threads should be cancelled at cancellation point
    EXPECT_LT(thread_completed.load(), thread_started.load())
        << "Some threads should be cancelled and not reach completion";
}

// Test 2: Cancel in parallel for loop
// Expected: Loop iterations stop when cancel is triggered
TEST_F(CancelTest, CancelParallelFor) {
    const int N = 1000;
    std::atomic<int> iterations_completed{0};
    std::atomic<int> cancel_triggered{0};
    const int target = 50;  // Earlier target to ensure it's reached

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        // Thread that processes iteration 'target' will cancel
        if (i == target) {
            cancel_triggered++;
            #pragma omp cancel for
        }

        // Check for cancellation before doing work
        #pragma omp cancellation point for

        // Do some work (only if not cancelled)
        iterations_completed++;
    }

    // We can't guarantee target iteration is executed (depends on scheduling)
    // But we CAN verify that if cancellation occurred, fewer iterations completed
    if (cancel_triggered.load() > 0) {
        EXPECT_LT(iterations_completed.load(), N)
            << "If cancellation was triggered, should complete fewer iterations";
    } else {
        // If target wasn't reached, all iterations should complete
        EXPECT_EQ(iterations_completed.load(), N)
            << "If no cancellation, all iterations should complete";
    }
}

// Test 3: Cancellation with if clause
// Expected: Cancel only activates when condition is true
TEST_F(CancelTest, CancelWithIfClause) {
    // Test with if(false) - cancellation should NOT activate
    std::atomic<int> completed_false{0};
    #pragma omp parallel num_threads(4)
    {
        #pragma omp cancel parallel if(false)
        #pragma omp cancellation point parallel
        completed_false++;
    }

    EXPECT_EQ(completed_false.load(), 4)
        << "With if(false), all threads should complete normally";

    // Test with if(true) - cancellation SHOULD activate
    std::atomic<int> started_true{0};
    std::atomic<int> completed_true{0};
    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();
        started_true++;

        if (tid == 0) {
            #pragma omp cancel parallel if(true)
            // Should not reach here
            completed_true++;
        } else {
            // Other threads check for cancellation
            #pragma omp cancellation point parallel
            // May or may not reach here depending on timing
            completed_true++;
        }
    }

    EXPECT_LT(completed_true.load(), started_true.load())
        << "With if(true), some threads should be cancelled";
}

// Test 4: Multiple cancellation points
// Expected: Threads should exit at the first encountered cancellation point
TEST_F(CancelTest, MultipleCancellationPoints) {
    std::atomic<int> reached_point1{0};
    std::atomic<int> reached_point2{0};
    std::atomic<int> reached_point3{0};
    std::atomic<int> cancel_count{0};

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        // Thread 0 requests cancellation and exits immediately
        if (tid == 0) {
            cancel_count++;
            #pragma omp cancel parallel
            // Should not reach any point below
            reached_point1++;
            reached_point2++;
            reached_point3++;
        } else {
            // Add some work to give Thread 0 time to set cancel flag
            volatile int sum = 0;
            for (int i = 0; i < 10000; i++) {
                sum += i;
            }

            // Other threads encounter first cancellation point
            #pragma omp cancellation point parallel
            reached_point1++;

            // Second cancellation point (should not reach if cancelled at point 1)
            #pragma omp cancellation point parallel
            reached_point2++;

            // Third cancellation point
            #pragma omp cancellation point parallel
            reached_point3++;
        }
    }

    // Thread 0 should have called cancel
    EXPECT_EQ(cancel_count.load(), 1);

    // Thread 0 should not reach any point
    // Other threads should be cancelled at first cancellation point
    EXPECT_LT(reached_point1.load(), 3) << "Most threads should exit at first cancellation point";
    EXPECT_LE(reached_point2.load(), reached_point1.load()) << "Fewer threads at point 2";
    EXPECT_LE(reached_point3.load(), reached_point2.load()) << "Fewer threads at point 3";
}

// Test 5: Barrier acts as implicit cancellation point (via __kmpc_cancel_barrier)
// Expected: When one thread cancels early, other threads detect cancellation at barrier
TEST_F(CancelTest, BarrierAsImplicitCancellationPoint) {
    std::atomic<int> after_barrier{0};

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        if (tid == 0) {
            // Thread 0 cancels immediately and exits
            #pragma omp cancel parallel
        } else {
            // Other threads do time-consuming work
            volatile int sum = 0;
            for (int i = 0; i < 1000000; i++) {
                sum += i;
            }
        }

        // Barrier should act as implicit cancellation point
        // Thread 0 has already exited, so only 3 threads reach here
        // But barrier will detect cancellation and those 3 threads should exit
        #pragma omp barrier

        // No thread should reach here
        after_barrier++;
    }

    EXPECT_EQ(after_barrier.load(), 0)
        << "No threads should pass barrier (barrier acts as cancellation point)";
}

// Test 6: Cancellation point after barrier
// Expected: Cancellation is checked at explicit cancellation point
TEST_F(CancelTest, CancellationPointAfterBarrier) {
    std::atomic<int> after_barrier{0};
    std::atomic<int> after_cancellation{0};

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        // All threads pass first barrier
        #pragma omp barrier
        after_barrier++;

        // Thread 0 cancels after barrier
        if (tid == 0) {
            #pragma omp cancel parallel
        }

        // Other threads check cancellation point
        #pragma omp cancellation point parallel
        after_cancellation++;
    }

    EXPECT_EQ(after_barrier.load(), 4) << "All threads should pass barrier";
    EXPECT_LT(after_cancellation.load(), 4)
        << "Some threads should be cancelled at cancellation point";
}

// Test 7: No cancellation point - threads complete normally
// Expected: Without cancellation points, cancel has no effect on execution
TEST_F(CancelTest, CancelWithoutCancellationPoint) {
    std::atomic<int> completed{0};

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        // Thread 0 requests cancellation
        if (tid == 0) {
            #pragma omp cancel parallel
            // Thread 0 itself should terminate after cancel
        } else {
            // Other threads have NO cancellation point
            // They should complete normally
            completed++;
        }
    }

    // Without cancellation points, non-cancelling threads complete normally
    EXPECT_EQ(completed.load(), 3)
        << "Threads without cancellation points should complete normally";
}

// Test 8: Correctness of computation despite cancellation
// Expected: Partial results before cancellation should be valid
TEST_F(CancelTest, ComputationCorrectnessWithCancel) {
    const int N = 1000;
    std::vector<int> result(N, 0);
    std::atomic<int> iterations_done{0};
    std::atomic<int> cancel_triggered{0};

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        // Trigger cancellation when we've done at least 100 iterations
        int current_count = iterations_done.load(std::memory_order_acquire);
        if (current_count >= 100 && cancel_triggered.load() == 0) {
            cancel_triggered++;
            #pragma omp cancel for
        }

        #pragma omp cancellation point for

        // Compute result
        result[i] = i * i;
        iterations_done++;
    }

    // Verify all processed elements have correct values
    for (int i = 0; i < N; i++) {
        if (result[i] != 0) {  // If processed
            EXPECT_EQ(result[i], i * i) << "Computed value should be correct at index " << i;
        }
    }

    // If cancellation was triggered, should not complete all iterations
    int final_count = iterations_done.load();
    if (cancel_triggered.load() > 0) {
        EXPECT_LT(final_count, N)
            << "Should not have processed all elements due to cancellation";
    }
}

// Test 9: omp_get_cancellation() API
// Expected: Returns true when cancellation is enabled
TEST_F(CancelTest, GetCancellationAPI) {
    int cancellation_enabled = omp_get_cancellation();

    // If we reach this test, SetUp() confirmed cancellation is enabled
    EXPECT_TRUE(cancellation_enabled)
        << "omp_get_cancellation() should return true when enabled";

    // Verify the value is consistent across parallel regions
    std::atomic<int> all_threads_see_enabled{0};

    #pragma omp parallel num_threads(4)
    {
        if (omp_get_cancellation()) {
            all_threads_see_enabled++;
        }
    }

    EXPECT_EQ(all_threads_see_enabled.load(), 4)
        << "All threads should see consistent cancellation state";
}

// Test 10: Cancel different construct types
// Expected: Cancel for loop only affects for loops, not parallel regions
TEST_F(CancelTest, CancelConstructTypeMatching) {
    std::atomic<int> for_completed{0};
    std::atomic<int> parallel_completed{0};

    #pragma omp parallel num_threads(2)
    {
        // Cancel for loop construct
        #pragma omp for
        for (int i = 0; i < 100; i++) {
            if (i == 10) {
                #pragma omp cancel for
            }
            #pragma omp cancellation point for
            for_completed++;
        }

        // Parallel region should NOT be affected by "cancel for"
        // Only checking for parallel cancellation
        #pragma omp cancellation point parallel
        parallel_completed++;
    }

    EXPECT_LT(for_completed.load(), 100)
        << "For loop should be cancelled early";
    EXPECT_EQ(parallel_completed.load(), 2)
        << "Parallel region should complete normally (no parallel cancel)";
}

// Test 11: Cancel with atomic operations
// Expected: Atomic operations before cancel should be visible
TEST_F(CancelTest, CancelWithAtomicOperations) {
    std::atomic<int> counter{0};
    std::atomic<int> cancel_count{0};
    const int N = 100;

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        // First increment counter
        counter++;

        // Cancel when counter reaches a threshold
        if (counter.load() >= 50 && cancel_count.load() == 0) {
            cancel_count++;  // Only one thread triggers cancel
            #pragma omp cancel for
        }

        #pragma omp cancellation point for
    }

    // If cancellation was triggered, should complete fewer than N iterations
    int final_count = counter.load();
    if (cancel_count.load() > 0) {
        EXPECT_LT(final_count, N)
            << "Should not complete all iterations if cancellation triggered";
    }

    // Counter should have been incremented multiple times (at least by some threads)
    EXPECT_GT(final_count, 0) << "Counter should have been incremented";
}
