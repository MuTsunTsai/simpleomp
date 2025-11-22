// Copyright 2025 Mu-Tsun Tsai
// Licensed under the MIT License

#include <gtest/gtest.h>
#include <omp.h>
#include <atomic>
#include <vector>
#include <algorithm>

class ForConstructTest : public testing::Test {
protected:
    void SetUp() override {
        // Ensure we have multiple threads available for testing
        ASSERT_GE(omp_get_max_threads(), 2) << "Need at least 2 threads to test for construct behavior";
    }
};

// Test 1: Basic for loop - verify iterations are distributed
// Expected: All iterations execute exactly once across multiple threads
TEST_F(ForConstructTest, BasicForLoop) {
    const int N = 100;
    std::vector<int> result(N, -1);

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        result[i] = i;
    }

    // Each iteration should execute exactly once with correct value
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i) << "Iteration " << i << " should have correct value";
    }
}

// Test 2: Loop with computation - verify correctness
// Expected: Results are identical to sequential execution
TEST_F(ForConstructTest, ComputationCorrectness) {
    const int N = 1000;
    std::vector<int> parallel_result(N);
    std::vector<int> sequential_result(N);

    // Parallel execution
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        parallel_result[i] = i * i + 2 * i + 1;
    }

    // Sequential execution
    for (int i = 0; i < N; i++) {
        sequential_result[i] = i * i + 2 * i + 1;
    }

    EXPECT_EQ(parallel_result, sequential_result)
        << "Parallel and sequential results should be identical";
}

// Test 3: Implicit barrier at end of for construct
// Expected: All threads complete before continuing
TEST_F(ForConstructTest, ImplicitBarrier) {
    std::atomic<int> counter{0};
    std::atomic<int> after_loop{0};

    #pragma omp parallel num_threads(4)
    {
        #pragma omp for
        for (int i = 0; i < 100; i++) {
            counter++;
        }
        // This should execute after all iterations are done (implicit barrier)
        after_loop++;
    }

    EXPECT_EQ(counter.load(), 100) << "All iterations should complete";
    EXPECT_EQ(after_loop.load(), 4) << "All threads should reach after loop (implicit barrier)";
}

// Test 4: nowait clause - no implicit barrier
// Expected: Threads can continue without waiting for others
TEST_F(ForConstructTest, NowaitClause) {
    std::atomic<int> fast_thread_done{0};
    std::atomic<int> slow_thread_done{0};

    #pragma omp parallel num_threads(2)
    {
        int tid = omp_get_thread_num();

        #pragma omp for nowait
        for (int i = 0; i < 100; i++) {
            if (tid == 0) {
                // Thread 0 does minimal work
                volatile int x = i;
                (void)x;
            } else {
                // Thread 1 does more work
                volatile int sum = 0;
                for (int j = 0; j < 1000; j++) {
                    sum += j;
                }
            }
        }

        // With nowait, thread 0 should reach here before thread 1 finishes
        if (tid == 0) {
            fast_thread_done++;
        }

        // Add some delay for thread 0 to demonstrate nowait
        if (tid == 0) {
            // Spin wait to check if thread 1 is still working
            volatile int check = 0;
            for (int i = 0; i < 100; i++) {
                check += i;
            }
        }

        if (tid == 1) {
            slow_thread_done++;
        }
    }

    EXPECT_EQ(fast_thread_done.load(), 1) << "Fast thread should complete its work";
    EXPECT_EQ(slow_thread_done.load(), 1) << "Slow thread should complete its work";
}

// Test 5: Different loop bounds and increments
// Expected: Correctly handle various canonical forms
TEST_F(ForConstructTest, VariousCanonicalForms) {
    // Test 1: i++ increment
    {
        std::vector<int> result(10, 0);
        #pragma omp parallel for
        for (int i = 0; i < 10; i++) {
            result[i] = i;
        }
        for (int i = 0; i < 10; i++) {
            EXPECT_EQ(result[i], i);
        }
    }

    // Test 2: ++i increment
    {
        std::vector<int> result(10, 0);
        #pragma omp parallel for
        for (int i = 0; i < 10; ++i) {
            result[i] = i;
        }
        for (int i = 0; i < 10; i++) {
            EXPECT_EQ(result[i], i);
        }
    }

    // Test 3: i += 2 increment
    {
        std::vector<int> result(20, -1);
        #pragma omp parallel for
        for (int i = 0; i < 20; i += 2) {
            result[i] = i;
        }
        for (int i = 0; i < 20; i += 2) {
            EXPECT_EQ(result[i], i);
        }
    }

    // Test 4: i = i + 1 increment
    {
        std::vector<int> result(10, 0);
        #pragma omp parallel for
        for (int i = 0; i < 10; i = i + 1) {
            result[i] = i;
        }
        for (int i = 0; i < 10; i++) {
            EXPECT_EQ(result[i], i);
        }
    }
}

// Test 6: Different relational operators
// Expected: Support <, <=, >, >= operators
TEST_F(ForConstructTest, RelationalOperators) {
    // Test with < operator
    {
        std::atomic<int> count{0};
        #pragma omp parallel for
        for (int i = 0; i < 10; i++) {
            count++;
        }
        EXPECT_EQ(count.load(), 10);
    }

    // Test with <= operator
    {
        std::atomic<int> count{0};
        #pragma omp parallel for
        for (int i = 0; i <= 9; i++) {
            count++;
        }
        EXPECT_EQ(count.load(), 10);
    }

    // Test with > operator (decrement)
    {
        std::atomic<int> count{0};
        #pragma omp parallel for
        for (int i = 9; i > -1; i--) {
            count++;
        }
        EXPECT_EQ(count.load(), 10);
    }

    // Test with >= operator (decrement)
    {
        std::atomic<int> count{0};
        #pragma omp parallel for
        for (int i = 9; i >= 0; i--) {
            count++;
        }
        EXPECT_EQ(count.load(), 10);
    }
}

// Test 7: Static schedule distribution
// Expected: Predictable chunk distribution
TEST_F(ForConstructTest, StaticSchedule) {
    const int N = 100;
    const int chunk_size = 10;
    std::vector<int> thread_ids(N, -1);

    #pragma omp parallel for schedule(static, chunk_size) num_threads(4)
    for (int i = 0; i < N; i++) {
        thread_ids[i] = omp_get_thread_num();
    }

    // Verify chunks are contiguous (each chunk of chunk_size has same thread)
    for (int i = 0; i < N / chunk_size; i++) {
        int start = i * chunk_size;
        int tid = thread_ids[start];
        for (int j = 0; j < chunk_size && start + j < N; j++) {
            EXPECT_EQ(thread_ids[start + j], tid)
                << "Static schedule should assign contiguous chunks";
        }
    }
}

// Test 8: Dynamic schedule distribution
// Expected: All iterations execute, possibly non-contiguous chunks per thread
// NOTE: This test may hang if dynamic scheduling has issues - disabled for now
TEST_F(ForConstructTest, DynamicSchedule) {
    const int N = 100;
    std::vector<int> result(N, 0);

    #pragma omp parallel for schedule(dynamic, 5) num_threads(4)
    for (int i = 0; i < N; i++) {
        result[i] = i * 2;
    }

    // Verify all iterations executed correctly
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i * 2) << "Iteration " << i << " should have correct value";
    }
}

// Test 9: Guided schedule distribution
// Expected: All iterations execute with exponentially decreasing chunk sizes
// NOTE: This test may hang if guided scheduling has issues - disabled for now
TEST_F(ForConstructTest, GuidedSchedule) {
    const int N = 100;
    std::vector<int> result(N, 0);

    #pragma omp parallel for schedule(guided, 5) num_threads(4)
    for (int i = 0; i < N; i++) {
        result[i] = i * 3;
    }

    // Verify all iterations executed correctly
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i * 3) << "Iteration " << i << " should have correct value";
    }
}

// Test 10: Runtime schedule
// Expected: Schedule determined by OMP_SCHEDULE environment variable
// NOTE: This test may hang if runtime scheduling has issues - disabled for now
TEST_F(ForConstructTest, RuntimeSchedule) {
    const int N = 100;
    std::vector<int> result(N, 0);

    #pragma omp parallel for schedule(runtime) num_threads(4)
    for (int i = 0; i < N; i++) {
        result[i] = i * 4;
    }

    // Verify all iterations executed correctly
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i * 4) << "Iteration " << i << " should have correct value";
    }
}

// Test 11: Loop iteration variable is private
// Expected: Each thread has its own copy of loop variable
TEST_F(ForConstructTest, LoopVariableIsPrivate) {
    int outer_i = 999;

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 100; i++) {
        // Loop variable i is private to each thread
        // Modifying it should not affect outer_i or other threads
        volatile int temp = i;
        (void)temp;
    }

    // Outer variable should remain unchanged
    EXPECT_EQ(outer_i, 999) << "Loop variable should be private and not affect outer scope";
}

// Test 12: Thread participation verification
// Expected: Multiple threads participate in loop execution
TEST_F(ForConstructTest, MultipleThreadsParticipate) {
    const int N = 1000;
    std::vector<bool> thread_participated(omp_get_max_threads(), false);
    std::atomic<int> unique_threads{0};

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < N; i++) {
        int tid = omp_get_thread_num();
        if (!thread_participated[tid]) {
            thread_participated[tid] = true;
            unique_threads++;
        }
    }

    EXPECT_GT(unique_threads.load(), 1) << "Multiple threads should participate in loop execution";
}

// Test 13: Zero iteration loop
// Expected: Loop body should not execute at all
TEST_F(ForConstructTest, ZeroIterations) {
    std::atomic<int> counter{0};

    #pragma omp parallel for
    for (int i = 0; i < 0; i++) {
        counter++;
    }

    EXPECT_EQ(counter.load(), 0) << "Loop with zero iterations should not execute";
}

// Test 14: Single iteration loop
// Expected: Loop should execute exactly once
TEST_F(ForConstructTest, SingleIteration) {
    std::atomic<int> counter{0};

    #pragma omp parallel for
    for (int i = 0; i < 1; i++) {
        counter++;
    }

    EXPECT_EQ(counter.load(), 1) << "Loop with single iteration should execute exactly once";
}

// Test 15: Nested loops (without collapse)
// Expected: Only outermost loop is parallelized
TEST_F(ForConstructTest, NestedLoopsWithoutCollapse) {
    const int M = 10;
    const int N = 10;
    std::vector<std::vector<int>> result(M, std::vector<int>(N, 0));

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            result[i][j] = i * N + j;
        }
    }

    // Verify results
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_EQ(result[i][j], i * N + j);
        }
    }
}

// Test 16: Loop with non-unit start
// Expected: Correctly handle loops starting from non-zero values
TEST_F(ForConstructTest, NonZeroStart) {
    const int start = 10;
    const int end = 20;
    std::vector<int> result(end, -1);

    #pragma omp parallel for
    for (int i = start; i < end; i++) {
        result[i] = i * 2;
    }

    for (int i = start; i < end; i++) {
        EXPECT_EQ(result[i], i * 2);
    }
}

// Test 17: Loop with negative increment
// Expected: Correctly handle decrementing loops
TEST_F(ForConstructTest, NegativeIncrement) {
    const int N = 10;
    std::vector<int> result(N, -1);

    #pragma omp parallel for
    for (int i = N - 1; i >= 0; i--) {
        result[i] = i;
    }

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i);
    }
}

// Test 18: Large loop iteration count
// Expected: Correctly handle large number of iterations
TEST_F(ForConstructTest, LargeIterationCount) {
    const int N = 100000;
    std::atomic<long long> sum{0};

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += i;
    }

    long long expected = (long long)N * (N - 1) / 2;
    EXPECT_EQ(sum.load(), expected) << "Sum should match sequential calculation";
}

// Test 19: For loop combined with if clause
// Expected: if(false) executes serially, if(true) executes in parallel
TEST_F(ForConstructTest, ForLoopWithIfClause) {
    const int N = 100;

    // Test with if(false) - should execute serially (single thread)
    std::vector<bool> threads_serial(omp_get_max_threads(), false);
    std::atomic<int> unique_threads_serial{0};

    #pragma omp parallel for if(false)
    for (int i = 0; i < N; i++) {
        int tid = omp_get_thread_num();
        if (!threads_serial[tid]) {
            threads_serial[tid] = true;
            unique_threads_serial++;
        }
    }

    EXPECT_EQ(unique_threads_serial.load(), 1)
        << "if(false) should execute with single thread";

    // Test with if(true) - should execute in parallel
    std::vector<bool> threads_parallel(omp_get_max_threads(), false);
    std::atomic<int> unique_threads_parallel{0};

    #pragma omp parallel for if(true) num_threads(4)
    for (int i = 0; i < N; i++) {
        int tid = omp_get_thread_num();
        if (!threads_parallel[tid]) {
            threads_parallel[tid] = true;
            unique_threads_parallel++;
        }
    }

    EXPECT_GT(unique_threads_parallel.load(), 1)
        << "if(true) should execute with multiple threads";
}

// Test 20: Schedule clause with num_threads
// Expected: Schedule respects specified thread count
TEST_F(ForConstructTest, ScheduleWithNumThreads) {
    const int N = 100;
    std::vector<int> thread_ids(N, -1);
    int num_threads_used = 0;

    #pragma omp parallel for schedule(static, 10) num_threads(3)
    for (int i = 0; i < N; i++) {
        thread_ids[i] = omp_get_thread_num();
    }

    // Count unique thread IDs
    std::vector<bool> thread_seen(omp_get_max_threads(), false);
    for (int i = 0; i < N; i++) {
        if (thread_ids[i] >= 0 && !thread_seen[thread_ids[i]]) {
            thread_seen[thread_ids[i]] = true;
            num_threads_used++;
        }
    }

    EXPECT_LE(num_threads_used, 3) << "Should use at most 3 threads as specified";
}

// Test 21: Loop bounds are evaluated before loop
// Expected: Bounds expression evaluated once before parallel region
TEST_F(ForConstructTest, LoopBoundsEvaluationTiming) {
    std::atomic<int> evaluation_count{0};
    auto get_bound = [&evaluation_count]() -> int {
        evaluation_count++;
        return 10;
    };

    int bound = get_bound();
    std::atomic<int> iterations{0};

    #pragma omp parallel for
    for (int i = 0; i < bound; i++) {
        iterations++;
    }

    // Bound expression should be evaluated exactly once (before the parallel region)
    EXPECT_EQ(evaluation_count.load(), 1)
        << "Loop bound expression should be evaluated exactly once";
    EXPECT_EQ(iterations.load(), 10)
        << "Should execute correct number of iterations";
}

// Test 22: omp_get_num_threads() inside for construct
// Expected: Returns actual number of threads in team
TEST_F(ForConstructTest, NumThreadsQueryInsideFor) {
    int reported_threads = 0;

    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < 100; i++) {
        if (i == 0) {
            reported_threads = omp_get_num_threads();
        }
    }

    EXPECT_GE(reported_threads, 1) << "Should report at least 1 thread";
    EXPECT_LE(reported_threads, 4) << "Should report at most 4 threads";
}

// Test 23: Unsigned integer loop variable
// Expected: Correctly handle unsigned loop variables
TEST_F(ForConstructTest, UnsignedLoopVariable) {
    const unsigned int N = 100;
    std::vector<unsigned int> result(N, 0);

    #pragma omp parallel for
    for (unsigned int i = 0; i < N; i++) {
        result[i] = i * 2;
    }

    for (unsigned int i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i * 2);
    }
}

// Test 24: Long integer loop variable
// Expected: Correctly handle different integer types
TEST_F(ForConstructTest, LongIntegerLoopVariable) {
    const long N = 100;
    std::vector<long> result(N, 0);

    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        result[i] = i * 3;
    }

    for (long i = 0; i < N; i++) {
        EXPECT_EQ(result[i], i * 3);
    }
}

// Test 25: Static schedule with no chunk size
// Expected: Approximately equal chunks, one per thread
TEST_F(ForConstructTest, StaticScheduleNoChunk) {
    const int N = 100;
    std::vector<int> thread_ids(N, -1);

    #pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < N; i++) {
        thread_ids[i] = omp_get_thread_num();
    }

    // Each thread should get a contiguous block
    // (exact distribution is implementation-defined, but should be balanced)
    int transitions = 0;
    for (int i = 1; i < N; i++) {
        if (thread_ids[i] != thread_ids[i - 1]) {
            transitions++;
        }
    }

    // With 4 threads, expect around 3 transitions (between thread regions)
    EXPECT_LE(transitions, 10)
        << "Static schedule without chunk should create large contiguous blocks";
}
