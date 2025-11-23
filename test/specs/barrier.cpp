// Copyright 2025 Mu-Tsun Tsai
// Licensed under the MIT License

#include <gtest/gtest.h>
#include <omp.h>
#include <atomic>
#include <vector>
#include <thread>
#include <chrono>

class BarrierTest : public testing::Test {
protected:
    void SetUp() override {
        // Ensure we have multiple threads available for testing
        ASSERT_GE(omp_get_max_threads(), 2) << "Need at least 2 threads to test barrier behavior";
    }
};

// Test 1: Basic barrier synchronization
// Expected: All threads wait at barrier until all arrive
TEST_F(BarrierTest, BasicBarrierSynchronization) {
    const int num_threads = 4;
    std::atomic<int> phase1_count{0};
    std::atomic<int> phase2_count{0};

    #pragma omp parallel num_threads(num_threads)
    {
        // Phase 1: Each thread increments counter
        phase1_count++;

        // Barrier: Wait for all threads
        #pragma omp barrier

        // Phase 2: All threads should see complete phase1_count
        int observed = phase1_count.load();
        EXPECT_EQ(observed, num_threads)
            << "Thread " << omp_get_thread_num()
            << " saw incomplete phase1_count";

        phase2_count++;
    }

    EXPECT_EQ(phase1_count.load(), num_threads);
    EXPECT_EQ(phase2_count.load(), num_threads);
}

// Test 2: Multiple sequential barriers
// Expected: Each barrier synchronizes independently
TEST_F(BarrierTest, MultipleSequentialBarriers) {
    const int num_threads = 4;
    const int num_phases = 3;
    std::vector<std::atomic<int>> phase_counts(num_phases);

    for (int i = 0; i < num_phases; i++) {
        phase_counts[i] = 0;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        for (int phase = 0; phase < num_phases; phase++) {
            // Each thread increments its phase counter
            phase_counts[phase]++;

            // Barrier between phases
            #pragma omp barrier

            // Verify all threads completed this phase
            EXPECT_EQ(phase_counts[phase].load(), num_threads)
                << "Phase " << phase << " incomplete at thread "
                << omp_get_thread_num();
        }
    }

    // Verify all phases completed
    for (int i = 0; i < num_phases; i++) {
        EXPECT_EQ(phase_counts[i].load(), num_threads)
            << "Phase " << i << " final count incorrect";
    }
}

// Test 3: Barrier with different work times
// Expected: Faster threads wait for slower threads
TEST_F(BarrierTest, BarrierWaitsForSlowerThreads) {
    const int num_threads = 4;
    std::atomic<int> started{0};
    std::atomic<int> finished{0};
    std::vector<bool> thread_finished(num_threads, false);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        started++;

        // Thread 0 sleeps longer to simulate slow work
        if (tid == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        // Mark this thread as finished with its work
        thread_finished[tid] = true;

        // Barrier: All threads must wait here
        #pragma omp barrier

        // After barrier, all threads should have finished their work
        for (int i = 0; i < num_threads; i++) {
            EXPECT_TRUE(thread_finished[i])
                << "Thread " << i << " not finished when thread "
                << tid << " passed barrier";
        }

        finished++;
    }

    EXPECT_EQ(started.load(), num_threads);
    EXPECT_EQ(finished.load(), num_threads);
}

// Test 4: Barrier ensures memory visibility
// Expected: Changes before barrier are visible to all threads after barrier
TEST_F(BarrierTest, BarrierEnsuresMemoryVisibility) {
    const int num_threads = 4;
    std::vector<int> shared_data(num_threads, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Each thread writes its thread ID to its position
        shared_data[tid] = tid + 1;

        // Barrier ensures all writes are visible
        #pragma omp barrier

        // After barrier, all threads should see all writes
        for (int i = 0; i < num_threads; i++) {
            EXPECT_EQ(shared_data[i], i + 1)
                << "Thread " << tid << " did not see write from thread " << i;
        }
    }
}

// Test 5: Barrier in conditional code
// Expected: All threads must encounter barrier or none at all
TEST_F(BarrierTest, BarrierInConditionalAllThreads) {
    const int num_threads = 4;
    std::atomic<int> before_barrier{0};
    std::atomic<int> after_barrier{0};
    bool condition = true;

    #pragma omp parallel num_threads(num_threads)
    {
        before_barrier++;

        // All threads take the same path
        if (condition) {
            #pragma omp barrier
        }

        after_barrier++;
    }

    EXPECT_EQ(before_barrier.load(), num_threads);
    EXPECT_EQ(after_barrier.load(), num_threads);
}

// Test 6: Single-threaded barrier (should be no-op)
// Expected: Barrier has no effect when only one thread
TEST_F(BarrierTest, SingleThreadedBarrierIsNoop) {
    int counter = 0;

    #pragma omp parallel num_threads(1)
    {
        EXPECT_EQ(omp_get_num_threads(), 1);
        counter++;

        #pragma omp barrier

        counter++;

        #pragma omp barrier

        counter++;
    }

    EXPECT_EQ(counter, 3);
}

// Test 7: Barrier with parallel for construct
// Expected: Implicit barrier at end of parallel for, then explicit barrier
TEST_F(BarrierTest, BarrierWithParallelFor) {
    const int num_threads = 4;
    const int array_size = 100;
    std::vector<int> data(array_size, 0);
    std::atomic<int> after_loop{0};
    std::atomic<int> after_barrier{0};

    #pragma omp parallel num_threads(num_threads)
    {
        // Parallel for has implicit barrier at end
        #pragma omp for
        for (int i = 0; i < array_size; i++) {
            data[i] = i;
        }

        // After implicit barrier, all iterations complete
        after_loop++;

        // Explicit barrier
        #pragma omp barrier

        // Verify all threads reached here
        after_barrier++;

        // Verify all data is written
        #pragma omp for
        for (int i = 0; i < array_size; i++) {
            EXPECT_EQ(data[i], i)
                << "Data not initialized at index " << i;
        }
    }

    EXPECT_EQ(after_loop.load(), num_threads);
    EXPECT_EQ(after_barrier.load(), num_threads);
}

// Test 8: Barrier ensures completion of all explicit tasks
// Expected: All tasks complete before threads pass barrier
// Note: SimpleOMP doesn't support explicit tasks, but we test basic behavior
TEST_F(BarrierTest, BarrierWithBasicWorkCompletion) {
    const int num_threads = 4;
    std::vector<std::atomic<bool>> work_done(num_threads);
    for (int i = 0; i < num_threads; i++) {
        work_done[i] = false;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Simulate task work
        work_done[tid] = true;

        // Barrier ensures all "tasks" are done
        #pragma omp barrier

        // Verify all work completed
        for (int i = 0; i < num_threads; i++) {
            EXPECT_TRUE(work_done[i].load())
                << "Thread " << i << " work not done when thread "
                << tid << " passed barrier";
        }
    }
}

// Test 9: Barrier ordering with worksharing constructs
// Expected: Sequence of worksharing and barrier regions must be same for all threads
TEST_F(BarrierTest, BarrierOrderingConsistency) {
    const int num_threads = 4;
    const int size = 100;
    std::vector<int> array1(size, 0);
    std::vector<int> array2(size, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        // First worksharing region
        #pragma omp for nowait
        for (int i = 0; i < size; i++) {
            array1[i] = i;
        }

        // Explicit barrier
        #pragma omp barrier

        // Second worksharing region
        #pragma omp for
        for (int i = 0; i < size; i++) {
            array2[i] = array1[i] * 2;
        }
    }

    // Verify results
    for (int i = 0; i < size; i++) {
        EXPECT_EQ(array1[i], i);
        EXPECT_EQ(array2[i], i * 2);
    }
}

// Test 10: Barrier is a task scheduling point
// Expected: Implementation may switch tasks at barrier
// Note: This test verifies barrier doesn't deadlock even with work imbalance
TEST_F(BarrierTest, BarrierIsTaskSchedulingPoint) {
    const int num_threads = 4;
    std::atomic<int> pre_barrier{0};
    std::atomic<int> post_barrier{0};

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Unbalanced work (thread 0 does more)
        if (tid == 0) {
            for (int i = 0; i < 1000; i++) {
                pre_barrier++;
            }
        } else {
            pre_barrier++;
        }

        // Barrier allows task scheduling
        #pragma omp barrier

        post_barrier++;
    }

    EXPECT_GT(pre_barrier.load(), num_threads);
    EXPECT_EQ(post_barrier.load(), num_threads);
}

// Test 11: Barrier with reduction-like pattern
// Expected: Barrier ensures all partial results are visible
TEST_F(BarrierTest, BarrierWithReductionPattern) {
    const int num_threads = 4;
    std::vector<int> partial_sums(num_threads, 0);
    int total_sum = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Each thread computes a partial sum
        partial_sums[tid] = tid * 10;

        // Barrier ensures all partial sums are ready
        #pragma omp barrier

        // Master thread aggregates results
        #pragma omp master
        {
            for (int i = 0; i < num_threads; i++) {
                total_sum += partial_sums[i];
            }
        }

        // Barrier ensures master finished aggregation
        #pragma omp barrier

        // All threads can now see the total
        EXPECT_EQ(total_sum, (num_threads - 1) * num_threads * 5)
            << "Thread " << tid << " sees incorrect total";
    }
}
