// MIT License
// Copyright 2025 Mu-Tsun Tsai

#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <atomic>
#include <algorithm>

// Test 1: Basic mutual exclusion - only one thread at a time in critical section
TEST(CriticalTest, BasicMutualExclusion) {
    const int num_threads = 4;
    std::atomic<int> concurrent_count(0);
    std::atomic<int> max_concurrent(0);
    int shared_counter = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp critical
        {
            // Track concurrent access
            int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;

            // Update max concurrent count
            int expected = max_concurrent.load(std::memory_order_relaxed);
            while (current > expected &&
                   !max_concurrent.compare_exchange_weak(expected, current,
                                                        std::memory_order_relaxed)) {
                // Retry if CAS failed
            }

            // Simulate some work
            volatile int dummy = 0;
            for (int i = 0; i < 100; i++) { dummy = dummy + 1; }

            // Non-atomic increment should be safe in critical section
            shared_counter++;

            concurrent_count.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    // Only one thread should be in critical section at any time
    EXPECT_EQ(max_concurrent.load(), 1);
    EXPECT_EQ(shared_counter, num_threads);
}

// Test 2: Named critical sections - different names are independent
TEST(CriticalTest, NamedCriticalIndependence) {
    const int num_threads = 4;
    std::atomic<int> section_a_concurrent(0);
    std::atomic<int> section_b_concurrent(0);
    std::atomic<int> max_a_concurrent(0);
    std::atomic<int> max_b_concurrent(0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        if (tid % 2 == 0) {
            #pragma omp critical(section_a)
            {
                int current = section_a_concurrent.fetch_add(1, std::memory_order_relaxed) + 1;
                int expected = max_a_concurrent.load(std::memory_order_relaxed);
                while (current > expected &&
                       !max_a_concurrent.compare_exchange_weak(expected, current)) {}

                volatile int dummy = 0;
                for (int i = 0; i < 1000; i++) { dummy = dummy + 1; }
                section_a_concurrent.fetch_sub(1, std::memory_order_relaxed);
            }
        } else {
            #pragma omp critical(section_b)
            {
                int current = section_b_concurrent.fetch_add(1, std::memory_order_relaxed) + 1;
                int expected = max_b_concurrent.load(std::memory_order_relaxed);
                while (current > expected &&
                       !max_b_concurrent.compare_exchange_weak(expected, current)) {}

                volatile int dummy = 0;
                for (int i = 0; i < 1000; i++) { dummy = dummy + 1; }
                section_b_concurrent.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    // Each named critical section should allow only one thread
    EXPECT_EQ(max_a_concurrent.load(), 1);
    EXPECT_EQ(max_b_concurrent.load(), 1);
}

// Test 3: Same name enforces mutual exclusion across all uses
TEST(CriticalTest, SameNameMutualExclusion) {
    const int num_threads = 4;
    std::atomic<int> concurrent_count(0);
    std::atomic<int> max_concurrent(0);
    int counter_a = 0;
    int counter_b = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        if (tid % 2 == 0) {
            #pragma omp critical(shared_lock)
            {
                int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
                int expected = max_concurrent.load(std::memory_order_relaxed);
                while (current > expected &&
                       !max_concurrent.compare_exchange_weak(expected, current)) {}

                counter_a++;
                volatile int dummy = 0;
                for (int i = 0; i < 100; i++) { dummy = dummy + 1; }
                concurrent_count.fetch_sub(1, std::memory_order_relaxed);
            }
        } else {
            #pragma omp critical(shared_lock)
            {
                int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
                int expected = max_concurrent.load(std::memory_order_relaxed);
                while (current > expected &&
                       !max_concurrent.compare_exchange_weak(expected, current)) {}

                counter_b++;
                volatile int dummy = 0;
                for (int i = 0; i < 100; i++) { dummy = dummy + 1; }
                concurrent_count.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    // All critical sections with same name should be mutually exclusive
    EXPECT_EQ(max_concurrent.load(), 1);
    EXPECT_EQ(counter_a + counter_b, num_threads);
}

// Test 4: Unnamed critical sections share the same lock
TEST(CriticalTest, UnnamedCriticalShared) {
    const int num_threads = 4;
    std::atomic<int> concurrent_count(0);
    std::atomic<int> max_concurrent(0);
    int counter_x = 0;
    int counter_y = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        if (tid % 2 == 0) {
            #pragma omp critical
            {
                int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
                int expected = max_concurrent.load(std::memory_order_relaxed);
                while (current > expected &&
                       !max_concurrent.compare_exchange_weak(expected, current)) {}

                counter_x++;
                volatile int dummy = 0;
                for (int i = 0; i < 100; i++) { dummy = dummy + 1; }
                concurrent_count.fetch_sub(1, std::memory_order_relaxed);
            }
        } else {
            #pragma omp critical
            {
                int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
                int expected = max_concurrent.load(std::memory_order_relaxed);
                while (current > expected &&
                       !max_concurrent.compare_exchange_weak(expected, current)) {}

                counter_y++;
                volatile int dummy = 0;
                for (int i = 0; i < 100; i++) { dummy = dummy + 1; }
                concurrent_count.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    // All unnamed critical sections should be mutually exclusive
    EXPECT_EQ(max_concurrent.load(), 1);
    EXPECT_EQ(counter_x + counter_y, num_threads);
}

// Test 5: Critical section execution order preservation
TEST(CriticalTest, ExecutionOrder) {
    const int num_threads = 4;
    const int iterations = 10;
    std::vector<int> execution_order;
    execution_order.reserve(num_threads * iterations);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        for (int i = 0; i < iterations; i++) {
            #pragma omp critical(order_test)
            {
                execution_order.push_back(tid);
            }
        }
    }

    // Total number of executions should match
    EXPECT_EQ(execution_order.size(), num_threads * iterations);

    // Each thread should have executed exactly 'iterations' times
    std::vector<int> thread_counts(num_threads, 0);
    for (int tid : execution_order) {
        thread_counts[tid]++;
    }

    for (int i = 0; i < num_threads; i++) {
        EXPECT_EQ(thread_counts[i], iterations);
    }
}

// Test 6: Critical section with complex data structure
TEST(CriticalTest, ComplexDataStructure) {
    const int num_threads = 4;
    const int items_per_thread = 100;
    std::vector<int> shared_vector;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        for (int i = 0; i < items_per_thread; i++) {
            #pragma omp critical
            {
                shared_vector.push_back(tid * items_per_thread + i);
            }
        }
    }

    // Verify all items were inserted
    EXPECT_EQ(shared_vector.size(), num_threads * items_per_thread);

    // Verify all values are unique (no race condition corruption)
    std::vector<int> sorted = shared_vector;
    std::sort(sorted.begin(), sorted.end());

    for (size_t i = 0; i < sorted.size(); i++) {
        if (i > 0) {
            EXPECT_NE(sorted[i], sorted[i-1]); // All values should be unique
        }
    }
}

// Test 7: Multiple critical sections in same parallel region
TEST(CriticalTest, MultipleCriticalSections) {
    const int num_threads = 4;
    int counter_alpha = 0;
    int counter_beta = 0;
    int counter_gamma = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp critical(alpha)
        {
            counter_alpha++;
        }

        #pragma omp critical(beta)
        {
            counter_beta++;
        }

        #pragma omp critical(gamma)
        {
            counter_gamma++;
        }
    }

    EXPECT_EQ(counter_alpha, num_threads);
    EXPECT_EQ(counter_beta, num_threads);
    EXPECT_EQ(counter_gamma, num_threads);
}

// Test 8: Critical section with conditional execution
TEST(CriticalTest, ConditionalExecution) {
    const int num_threads = 8;
    int even_counter = 0;
    int odd_counter = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        if (tid % 2 == 0) {
            #pragma omp critical(even_section)
            {
                even_counter++;
            }
        } else {
            #pragma omp critical(odd_section)
            {
                odd_counter++;
            }
        }
    }

    EXPECT_EQ(even_counter, num_threads / 2);
    EXPECT_EQ(odd_counter, num_threads / 2);
}

// Test 9: Critical section protects read-modify-write operations
TEST(CriticalTest, ReadModifyWrite) {
    const int num_threads = 4;
    const int iterations = 1000;
    int shared_value = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        for (int i = 0; i < iterations; i++) {
            #pragma omp critical
            {
                // Read-modify-write sequence
                int temp = shared_value;
                temp = temp + 1;
                shared_value = temp;
            }
        }
    }

    EXPECT_EQ(shared_value, num_threads * iterations);
}

// Test 10: Nested parallel regions with critical sections
// Note: SimpleOMP doesn't support nested parallelism, so this tests
// critical sections in sequential nested regions
TEST(CriticalTest, CriticalInNestedContext) {
    const int outer_threads = 4;
    int outer_counter = 0;

    #pragma omp parallel num_threads(outer_threads)
    {
        #pragma omp critical(outer)
        {
            outer_counter++;

            // Inner parallel region will execute sequentially (no nesting support)
            int inner_counter = 0;
            #pragma omp parallel num_threads(2)
            {
                #pragma omp critical(inner)
                {
                    inner_counter++;
                }
            }

            // Inner region should have executed sequentially
            EXPECT_TRUE(inner_counter > 0);
        }
    }

    EXPECT_EQ(outer_counter, outer_threads);
}

// Test 11: Critical section ensures visibility of writes
TEST(CriticalTest, MemoryVisibility) {
    const int num_threads = 4;
    int shared_data[num_threads];
    bool ready_flags[num_threads];

    // Initialize
    for (int i = 0; i < num_threads; i++) {
        shared_data[i] = 0;
        ready_flags[i] = false;
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Write to own slot
        shared_data[tid] = tid + 1;

        // Mark as ready in critical section
        #pragma omp critical
        {
            ready_flags[tid] = true;
        }

        // Barrier to ensure all threads have written
        #pragma omp barrier

        // Read others' data in critical section
        #pragma omp critical
        {
            for (int i = 0; i < num_threads; i++) {
                if (ready_flags[i]) {
                    EXPECT_EQ(shared_data[i], i + 1);
                }
            }
        }
    }
}

// Test 12: Critical section with conditional execution inside
TEST(CriticalTest, ConditionalInCritical) {
    const int num_threads = 4;
    std::atomic<int> execution_count(0);
    std::atomic<int> conditional_count(0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        #pragma omp critical
        {
            execution_count.fetch_add(1, std::memory_order_relaxed);

            if (tid != 2) {
                // Only some threads execute this
                conditional_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    // All 4 threads enter critical section
    EXPECT_EQ(execution_count.load(), 4);
    // Only 3 threads execute conditional code (tid != 2)
    EXPECT_EQ(conditional_count.load(), 3);
}

// Test 13: Critical section fairness (all threads eventually get access)
TEST(CriticalTest, Fairness) {
    const int num_threads = 8;
    std::atomic<bool> thread_executed[num_threads];

    for (int i = 0; i < num_threads; i++) {
        thread_executed[i].store(false, std::memory_order_relaxed);
    }

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();

        // Spin until we get into critical section
        bool entered = false;
        while (!entered) {
            #pragma omp critical(fairness_test)
            {
                thread_executed[tid].store(true, std::memory_order_relaxed);
                entered = true;
            }
        }
    }

    // All threads should have executed
    for (int i = 0; i < num_threads; i++) {
        EXPECT_TRUE(thread_executed[i].load(std::memory_order_relaxed));
    }
}

// Test 14: Critical section with barrier interaction
TEST(CriticalTest, CriticalWithBarrier) {
    const int num_threads = 4;
    int phase1_counter = 0;
    int phase2_counter = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        // Phase 1: All threads increment counter
        #pragma omp critical
        {
            phase1_counter++;
        }

        // Synchronization point
        #pragma omp barrier

        // Phase 2: All threads can see phase1 result
        #pragma omp critical
        {
            EXPECT_EQ(phase1_counter, num_threads);
            phase2_counter++;
        }
    }

    EXPECT_EQ(phase1_counter, num_threads);
    EXPECT_EQ(phase2_counter, num_threads);
}

// Test 15: Critical section performance (no false sharing)
TEST(CriticalTest, Performance) {
    const int num_threads = 4;
    const int iterations = 10000;
    int shared_counter = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        for (int i = 0; i < iterations; i++) {
            #pragma omp critical
            {
                shared_counter++;
            }
        }
    }

    EXPECT_EQ(shared_counter, num_threads * iterations);
}
