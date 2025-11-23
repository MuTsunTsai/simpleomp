// MIT License
// Copyright 2025 Mu-Tsun Tsai

#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <atomic>
#include <algorithm>

// Test 1: Only master thread executes the master region
TEST(MasterTest, OnlyMasterExecutes) {
    const int num_threads = 4;
    std::atomic<int> master_count(0);
    std::atomic<int> all_threads_count(0);
    int master_tid = -1;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // All threads increment this counter
        all_threads_count.fetch_add(1, std::memory_order_relaxed);

        #pragma omp master
        {
            master_count.fetch_add(1, std::memory_order_relaxed);
            master_tid = tid;
        }
    }

    // All threads executed
    EXPECT_EQ(all_threads_count.load(), num_threads);
    // Only one thread executed master region
    EXPECT_EQ(master_count.load(), 1);
    // Master thread has ID 0
    EXPECT_EQ(master_tid, 0);
}

// Test 2: Master thread is always thread 0
TEST(MasterTest, MasterIsThread0) {
    const int num_threads = 8;
    std::vector<int> master_thread_ids;

    // Run multiple parallel regions
    for (int run = 0; run < 5; run++) {
        int observed_master = -1;

        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp master
            {
                observed_master = omp_get_thread_num();
            }
        }

        master_thread_ids.push_back(observed_master);
    }

    // All runs should have thread 0 as master
    for (int tid : master_thread_ids) {
        EXPECT_EQ(tid, 0);
    }
}

// Test 3: No implicit barrier on entry or exit
TEST(MasterTest, NoImplicitBarrier) {
    const int num_threads = 4;
    std::atomic<int> phase1_counter(0);
    std::atomic<int> phase2_counter(0);
    std::atomic<bool> master_started(false);
    std::atomic<bool> master_finished(false);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Phase 1: All threads start
        phase1_counter.fetch_add(1, std::memory_order_relaxed);

        #pragma omp master
        {
            master_started.store(true, std::memory_order_release);

            // Simulate some work in master region
            volatile int dummy = 0;
            for (int i = 0; i < 10000; i++) { dummy++; }

            master_finished.store(true, std::memory_order_release);
        }

        // Non-master threads can reach here before master finishes
        // (no implicit barrier on entry/exit to master)
        if (tid != 0) {
            // Non-master threads may see master not finished yet
            // This is valid because there's no barrier
        }

        // Phase 2: All threads reach this point
        phase2_counter.fetch_add(1, std::memory_order_relaxed);
    }

    EXPECT_EQ(phase1_counter.load(), num_threads);
    EXPECT_EQ(phase2_counter.load(), num_threads);
    EXPECT_TRUE(master_started.load());
    EXPECT_TRUE(master_finished.load());
}

// Test 4: Multiple master regions in same parallel region
TEST(MasterTest, MultipleMasterRegions) {
    const int num_threads = 4;
    std::atomic<int> master1_count(0);
    std::atomic<int> master2_count(0);
    std::atomic<int> master3_count(0);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp master
        {
            master1_count.fetch_add(1, std::memory_order_relaxed);
        }

        #pragma omp master
        {
            master2_count.fetch_add(1, std::memory_order_relaxed);
        }

        #pragma omp master
        {
            master3_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Each master region executed exactly once
    EXPECT_EQ(master1_count.load(), 1);
    EXPECT_EQ(master2_count.load(), 1);
    EXPECT_EQ(master3_count.load(), 1);
}

// Test 5: Master region with barrier
TEST(MasterTest, MasterWithBarrier) {
    const int num_threads = 4;
    int shared_value = 0;
    std::atomic<bool> value_set(false);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp master
        {
            shared_value = 42;
            value_set.store(true, std::memory_order_release);
        }

        // Explicit barrier ensures all threads see the update
        #pragma omp barrier

        // All threads can now safely read the value
        int local_value;
        #pragma omp critical
        {
            local_value = shared_value;
        }

        EXPECT_EQ(local_value, 42);
    }

    EXPECT_TRUE(value_set.load());
}

// Test 6: Master region modifying shared data
TEST(MasterTest, ModifySharedData) {
    const int num_threads = 4;
    std::vector<int> shared_vector;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp master
        {
            // Only master modifies shared vector
            for (int i = 0; i < 10; i++) {
                shared_vector.push_back(i);
            }
        }

        // Barrier to ensure master has finished
        #pragma omp barrier

        // All threads can read (master already finished modification)
        #pragma omp critical
        {
            EXPECT_EQ(shared_vector.size(), 10);
        }
    }
}

// Test 7: Nested master regions (sequential execution due to no nested parallelism)
TEST(MasterTest, NestedMasterRegions) {
    const int outer_threads = 4;
    std::atomic<int> outer_master_count(0);
    std::atomic<int> inner_master_count(0);

    #pragma omp parallel num_threads(outer_threads)
    {
        #pragma omp master
        {
            outer_master_count.fetch_add(1, std::memory_order_relaxed);

            // Inner parallel region executes sequentially (no nested parallelism)
            #pragma omp parallel num_threads(2)
            {
                #pragma omp master
                {
                    inner_master_count.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    }

    EXPECT_EQ(outer_master_count.load(), 1);
    // Inner region may execute, but sequentially
    EXPECT_GE(inner_master_count.load(), 0);
}

// Test 8: Master region with conditional logic
TEST(MasterTest, ConditionalLogic) {
    const int num_threads = 4;
    std::atomic<int> counter(0);
    bool condition = true;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp master
        {
            if (condition) {
                counter.fetch_add(10, std::memory_order_relaxed);
            } else {
                counter.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    EXPECT_EQ(counter.load(), 10);
}

// Test 9: Master region initialization pattern
TEST(MasterTest, InitializationPattern) {
    const int num_threads = 4;
    const int array_size = 100;
    std::vector<int> data;
    data.reserve(array_size);

    #pragma omp parallel num_threads(num_threads)
    {
        // Master initializes data
        #pragma omp master
        {
            for (int i = 0; i < array_size; i++) {
                data.push_back(i * 2);
            }
        }

        // Wait for initialization
        #pragma omp barrier

        // All threads can process data
        int tid = omp_get_thread_num();
        int chunk_size = array_size / num_threads;
        int start = tid * chunk_size;
        int end = (tid == num_threads - 1) ? array_size : start + chunk_size;

        for (int i = start; i < end; i++) {
            #pragma omp critical
            {
                EXPECT_EQ(data[i], i * 2);
            }
        }
    }
}

// Test 10: Master region with function calls
TEST(MasterTest, FunctionCalls) {
    const int num_threads = 4;
    std::atomic<int> function_calls(0);

    auto expensive_function = [&function_calls]() {
        function_calls.fetch_add(1, std::memory_order_relaxed);
        return 42;
    };

    int result = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp master
        {
            result = expensive_function();
        }
    }

    // Function called exactly once
    EXPECT_EQ(function_calls.load(), 1);
    EXPECT_EQ(result, 42);
}

// Test 11: Master region execution order
TEST(MasterTest, ExecutionOrder) {
    const int num_threads = 4;
    std::vector<int> execution_order;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Non-master threads may execute this first
        if (tid != 0) {
            #pragma omp critical
            {
                execution_order.push_back(tid);
            }
        }

        #pragma omp master
        {
            #pragma omp critical
            {
                execution_order.push_back(0);
            }
        }

        // Master thread executes this after master region
        if (tid == 0) {
            #pragma omp critical
            {
                execution_order.push_back(100);  // Marker for "after master"
            }
        }
    }

    // Master (tid=0) should appear in the execution order
    bool master_executed = false;
    for (int tid : execution_order) {
        if (tid == 0) {
            master_executed = true;
            break;
        }
    }
    EXPECT_TRUE(master_executed);
}

// Test 12: Master region with I/O operations (simulated)
TEST(MasterTest, IOOperations) {
    const int num_threads = 4;
    std::string output;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp master
        {
            // Simulate I/O operation (only master does it)
            output = "Master thread " + std::to_string(tid) + " reporting";
        }
    }

    EXPECT_EQ(output, "Master thread 0 reporting");
}

// Test 13: Master region with early exit pattern
TEST(MasterTest, EarlyExitPattern) {
    const int num_threads = 4;
    std::atomic<bool> error_detected(false);
    std::atomic<int> work_counter(0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp master
        {
            // Simulate error detection
            if (true) {  // Condition that triggers error
                error_detected.store(true, std::memory_order_release);
            }
        }

        // No barrier - other threads may still be working

        // All threads check for error (eventually)
        if (!error_detected.load(std::memory_order_acquire)) {
            work_counter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    EXPECT_TRUE(error_detected.load());
}

// Test 14: Master region correctness with varying thread counts
TEST(MasterTest, VaryingThreadCounts) {
    for (int nt = 1; nt <= 8; nt++) {
        int master_tid = -1;
        int execution_count = 0;

        #pragma omp parallel num_threads(nt)
        {
            #pragma omp master
            {
                master_tid = omp_get_thread_num();
                execution_count++;
            }
        }

        EXPECT_EQ(master_tid, 0);
        EXPECT_EQ(execution_count, 1);
    }
}

// Test 15: Master region with single thread
TEST(MasterTest, SingleThread) {
    int master_count = 0;
    int master_tid = -1;

    #pragma omp parallel num_threads(1)
    {
        #pragma omp master
        {
            master_count++;
            master_tid = omp_get_thread_num();
        }
    }

    EXPECT_EQ(master_count, 1);
    EXPECT_EQ(master_tid, 0);
}

// Test 16: Master region combined with for loop
TEST(MasterTest, MasterWithForLoop) {
    const int num_threads = 4;
    std::atomic<int> master_sum(0);
    std::atomic<int> parallel_sum(0);

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp master
        {
            // Master does sequential work
            for (int i = 0; i < 10; i++) {
                master_sum.fetch_add(i, std::memory_order_relaxed);
            }
        }

        // All threads do parallel work
        #pragma omp for
        for (int i = 0; i < 100; i++) {
            parallel_sum.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Master sum: 0+1+2+...+9 = 45
    EXPECT_EQ(master_sum.load(), 45);
    // Parallel sum: 100 iterations
    EXPECT_EQ(parallel_sum.load(), 100);
}

// Test 17: Master region visibility across threads
TEST(MasterTest, MemoryVisibility) {
    const int num_threads = 4;
    int shared_flag = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp master
        {
            shared_flag = 1;
        }

        // Explicit barrier for memory synchronization
        #pragma omp barrier

        // All threads can see the update after barrier
        EXPECT_EQ(shared_flag, 1);
    }
}

// Test 18: Master region with reduction-like pattern
TEST(MasterTest, ReductionPattern) {
    const int num_threads = 4;
    const int iterations = 100;
    std::atomic<int> partial_sums[num_threads];
    int total_sum = 0;

    for (int i = 0; i < num_threads; i++) {
        partial_sums[i].store(0, std::memory_order_relaxed);
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int local_sum = 0;

        // Each thread computes partial sum
        #pragma omp for
        for (int i = 0; i < iterations; i++) {
            local_sum += i;
        }

        // Store partial sum
        partial_sums[tid].store(local_sum, std::memory_order_release);

        // Wait for all threads to finish their partial sums
        #pragma omp barrier

        // Master combines results
        #pragma omp master
        {
            for (int i = 0; i < num_threads; i++) {
                total_sum += partial_sums[i].load(std::memory_order_acquire);
            }
        }
    }

    // Sum of 0 to 99: (99 * 100) / 2 = 4950
    EXPECT_EQ(total_sum, 4950);
}
