// Copyright 2025 Mu-Tsun Tsai
// Licensed under the MIT License

#include <gtest/gtest.h>
#include <omp.h>
#include <atomic>
#include <vector>
#include <cmath>
#include <limits>

class AtomicTest : public testing::Test {
protected:
    void SetUp() override {
        // Ensure we have multiple threads available for testing
        ASSERT_GE(omp_get_max_threads(), 2) << "Need at least 2 threads to test atomic operations";
        omp_set_num_threads(4);
    }
};

//==============================================================================
// Test 1: Basic atomic update - increment operations
//==============================================================================
TEST_F(AtomicTest, BasicIncrementOperations) {
    int counter1 = 0;
    int counter2 = 0;
    int counter3 = 0;
    int counter4 = 0;
    const int iterations = 1000;

    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
        #pragma omp atomic
        counter1++;

        #pragma omp atomic
        ++counter2;

        #pragma omp atomic
        counter3--;

        #pragma omp atomic
        --counter4;
    }

    EXPECT_EQ(counter1, iterations) << "x++ should increment correctly";
    EXPECT_EQ(counter2, iterations) << "++x should increment correctly";
    EXPECT_EQ(counter3, -iterations) << "x-- should decrement correctly";
    EXPECT_EQ(counter4, -iterations) << "--x should decrement correctly";
}

//==============================================================================
// Test 2: Atomic update - addition operations
//==============================================================================
TEST_F(AtomicTest, AtomicAddition) {
    int sum_int = 0;
    float sum_float = 0.0f;
    double sum_double = 0.0;
    const int N = 1000;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        sum_int += 1;

        #pragma omp atomic
        sum_float += 1.0f;

        #pragma omp atomic
        sum_double += 1.0;
    }

    EXPECT_EQ(sum_int, N) << "Integer addition should be atomic";
    EXPECT_FLOAT_EQ(sum_float, static_cast<float>(N)) << "Float addition should be atomic";
    EXPECT_DOUBLE_EQ(sum_double, static_cast<double>(N)) << "Double addition should be atomic";
}

//==============================================================================
// Test 3: Atomic update - subtraction operations
//==============================================================================
TEST_F(AtomicTest, AtomicSubtraction) {
    int diff = 10000;
    float diff_float = 10000.0f;
    double diff_double = 10000.0;
    const int N = 1000;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        diff -= 1;

        #pragma omp atomic
        diff_float -= 1.0f;

        #pragma omp atomic
        diff_double -= 1.0;
    }

    EXPECT_EQ(diff, 10000 - N) << "Integer subtraction should be atomic";
    EXPECT_FLOAT_EQ(diff_float, 10000.0f - N) << "Float subtraction should be atomic";
    EXPECT_DOUBLE_EQ(diff_double, 10000.0 - N) << "Double subtraction should be atomic";
}

//==============================================================================
// Test 4: Atomic update - multiplication operations
//==============================================================================
TEST_F(AtomicTest, AtomicMultiplication) {
    int product = 1;
    float product_float = 1.0f;
    const int iterations = 10;

    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
        #pragma omp atomic
        product *= 2;

        #pragma omp atomic
        product_float *= 2.0f;
    }

    EXPECT_EQ(product, 1 << iterations) << "Integer multiplication should be atomic";
    EXPECT_FLOAT_EQ(product_float, static_cast<float>(1 << iterations))
        << "Float multiplication should be atomic";
}

//==============================================================================
// Test 5: Atomic update - division operations
//==============================================================================
TEST_F(AtomicTest, AtomicDivision) {
    int quotient = 1024;
    double quotient_double = 1024.0;
    const int iterations = 10;

    #pragma omp parallel for
    for (int i = 0; i < iterations; i++) {
        #pragma omp atomic
        quotient /= 2;

        #pragma omp atomic
        quotient_double /= 2.0;
    }

    EXPECT_EQ(quotient, 1) << "Integer division should be atomic";
    EXPECT_DOUBLE_EQ(quotient_double, 1.0) << "Double division should be atomic";
}

//==============================================================================
// Test 6: Atomic update - bitwise AND operations
//==============================================================================
TEST_F(AtomicTest, AtomicBitwiseAND) {
    unsigned int mask = 0xFFFFFFFF;
    const int N = 32;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        unsigned int bit_mask = ~(1u << (i % 32));
        #pragma omp atomic
        mask &= bit_mask;
    }

    EXPECT_EQ(mask, 0u) << "Bitwise AND should be atomic";
}

//==============================================================================
// Test 7: Atomic update - bitwise OR operations
//==============================================================================
TEST_F(AtomicTest, AtomicBitwiseOR) {
    unsigned int flags = 0;
    const int N = 32;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        unsigned int bit_flag = 1u << (i % 32);
        #pragma omp atomic
        flags |= bit_flag;
    }

    EXPECT_EQ(flags, 0xFFFFFFFFu) << "Bitwise OR should be atomic";
}

//==============================================================================
// Test 8: Atomic update - bitwise XOR operations
//==============================================================================
TEST_F(AtomicTest, AtomicBitwiseXOR) {
    unsigned int value = 0;
    const unsigned int pattern = 0xAAAAAAAA;
    const int N = 1000;

    // XOR with same value twice should cancel out
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        value ^= pattern;

        #pragma omp atomic
        value ^= pattern;
    }

    EXPECT_EQ(value, 0u) << "Bitwise XOR should be atomic";
}

//==============================================================================
// Test 9: Atomic min operations using critical (atomic min/max not in standard form)
//==============================================================================
TEST_F(AtomicTest, MinMaxWithCritical) {
    int min_int = 1000;
    int max_int = 0;
    float min_float = 1000.0f;
    float max_float = 0.0f;

    #pragma omp parallel for
    for (int i = 0; i < 1000; i++) {
        #pragma omp critical
        {
            if (i < min_int) min_int = i;
            if (i > max_int) max_int = i;
            if (static_cast<float>(i) < min_float) min_float = static_cast<float>(i);
            if (static_cast<float>(i) > max_float) max_float = static_cast<float>(i);
        }
    }

    EXPECT_EQ(min_int, 0) << "Min using critical should find minimum";
    EXPECT_EQ(max_int, 999) << "Max using critical should find maximum";
    EXPECT_FLOAT_EQ(min_float, 0.0f) << "Float min using critical should find minimum";
    EXPECT_FLOAT_EQ(max_float, 999.0f) << "Float max using critical should find maximum";
}

//==============================================================================
// Test 10: Atomic read operations
//==============================================================================
TEST_F(AtomicTest, AtomicRead) {
    int shared_value = 42;
    const int num_threads = 4;
    std::vector<int> read_values(num_threads, 0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        int local_value;

        #pragma omp atomic read
        local_value = shared_value;

        read_values[tid] = local_value;
    }

    // All threads should read the same value
    for (int i = 0; i < num_threads; i++) {
        EXPECT_EQ(read_values[i], 42) << "Atomic read should return consistent value for thread " << i;
    }
}

//==============================================================================
// Test 11: Atomic write operations
//==============================================================================
TEST_F(AtomicTest, AtomicWrite) {
    int shared_value = 0;
    const int N = 100;
    std::atomic<int> write_count{0};

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic write
        shared_value = i;
        write_count++;
    }

    EXPECT_EQ(write_count.load(), N) << "All threads should complete their writes";
    // The final value depends on thread scheduling, but should be in valid range
    EXPECT_GE(shared_value, 0);
    EXPECT_LT(shared_value, N);
}

//==============================================================================
// Test 12: Multiple atomic operations on different variables
//==============================================================================
TEST_F(AtomicTest, MultipleAtomicVariables) {
    int sum1 = 0, sum2 = 0, sum3 = 0;
    const int N = 1000;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        sum1 += i;

        #pragma omp atomic
        sum2 += i * 2;

        #pragma omp atomic
        sum3 += i * 3;
    }

    int expected_sum = (N - 1) * N / 2;
    EXPECT_EQ(sum1, expected_sum);
    EXPECT_EQ(sum2, expected_sum * 2);
    EXPECT_EQ(sum3, expected_sum * 3);
}

//==============================================================================
// Test 13: Atomic operations with different integer sizes
//==============================================================================
TEST_F(AtomicTest, DifferentIntegerSizes) {
    int8_t counter8 = 0;
    int16_t counter16 = 0;
    int32_t counter32 = 0;
    int64_t counter64 = 0;
    const int N = 100;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        counter8++;

        #pragma omp atomic
        counter16++;

        #pragma omp atomic
        counter32++;

        #pragma omp atomic
        counter64++;
    }

    EXPECT_EQ(counter8, static_cast<int8_t>(N));
    EXPECT_EQ(counter16, static_cast<int16_t>(N));
    EXPECT_EQ(counter32, static_cast<int32_t>(N));
    EXPECT_EQ(counter64, static_cast<int64_t>(N));
}

//==============================================================================
// Test 14: Atomic operations with unsigned integers
//==============================================================================
TEST_F(AtomicTest, UnsignedIntegerOperations) {
    uint32_t unsigned_sum = 0;
    uint64_t unsigned_product = 1;
    const int N = 100;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        unsigned_sum += static_cast<uint32_t>(i);

        if (i < 10) {
            #pragma omp atomic
            unsigned_product *= 2;
        }
    }

    uint32_t expected_sum = static_cast<uint32_t>((N - 1) * N / 2);
    EXPECT_EQ(unsigned_sum, expected_sum);
    EXPECT_EQ(unsigned_product, static_cast<uint64_t>(1 << 10));
}

//==============================================================================
// Test 15: Atomic operations - alternative syntax forms
//==============================================================================
TEST_F(AtomicTest, AlternativeSyntaxForms) {
    int x = 0;
    const int N = 1000;

    // Test: x = x binop expr
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        x = x + 1;
    }
    EXPECT_EQ(x, N) << "x = x + expr form should work";

    // Test: x = expr binop x
    x = 0;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        x = 1 + x;
    }
    EXPECT_EQ(x, N) << "x = expr + x form should work";

    // Test: x binop= expr
    x = 0;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        x += 1;
    }
    EXPECT_EQ(x, N) << "x += expr form should work";
}

//==============================================================================
// Test 16: Atomic operations prevent data races
//==============================================================================
TEST_F(AtomicTest, DataRacePrevention) {
    const int N = 10000;
    int atomic_counter = 0;
    int non_atomic_counter = 0;

    // Atomic version - should always get correct result
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        atomic_counter++;
    }

    EXPECT_EQ(atomic_counter, N) << "Atomic operations should prevent data races";

    // Non-atomic version - may have race conditions (for comparison)
    // Note: We can't reliably test that non-atomic FAILS, but we document the difference
    std::atomic<int> safe_counter{0};
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        safe_counter++;  // Using std::atomic for testing purposes
    }
    EXPECT_EQ(safe_counter.load(), N);
}

//==============================================================================
// Test 17: Atomic with complex expressions
//==============================================================================
TEST_F(AtomicTest, ComplexExpressions) {
    int result = 0;
    const int N = 100;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        // Expression expr should not access x's storage
        int local_value = i * 2 + 5;
        #pragma omp atomic
        result += local_value;
    }

    int expected = 0;
    for (int i = 0; i < N; i++) {
        expected += i * 2 + 5;
    }
    EXPECT_EQ(result, expected);
}

//==============================================================================
// Test 18: Atomic operations with floating-point edge cases
//==============================================================================
TEST_F(AtomicTest, FloatingPointEdgeCases) {
    float f_value = 0.0f;
    double d_value = 0.0;

    // Test with small increments
    const int N = 1000;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        f_value += 0.1f;

        #pragma omp atomic
        d_value += 0.1;
    }

    EXPECT_NEAR(f_value, N * 0.1f, 0.01f) << "Float atomic should handle small increments";
    EXPECT_NEAR(d_value, N * 0.1, 0.001) << "Double atomic should handle small increments";
}

//==============================================================================
// Test 19: Atomic update with binop= operator precedence
//==============================================================================
TEST_F(AtomicTest, OperatorPrecedence) {
    int x = 100;
    const int N = 10;

    // Test multiplication precedence
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        x += 2 * 3;  // Should be x += (2 * 3), not (x += 2) * 3
    }

    EXPECT_EQ(x, 100 + N * 6) << "Operator precedence should be respected";
}

//==============================================================================
// Test 20: Stress test - high contention scenario
//==============================================================================
TEST_F(AtomicTest, HighContentionStressTest) {
    int shared_counter = 0;
    const int iterations_per_thread = 10000;
    const int num_threads = 4;

    #pragma omp parallel num_threads(num_threads)
    {
        for (int i = 0; i < iterations_per_thread; i++) {
            #pragma omp atomic
            shared_counter++;
        }
    }

    EXPECT_EQ(shared_counter, num_threads * iterations_per_thread)
        << "Atomic should handle high contention correctly";
}

//==============================================================================
// Test 21: Atomic operations with mixed types in parallel region
//==============================================================================
TEST_F(AtomicTest, MixedTypesInParallelRegion) {
    int int_sum = 0;
    long long_sum = 0L;
    float float_sum = 0.0f;
    double double_sum = 0.0;
    const int N = 1000;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        int_sum += 1;

        #pragma omp atomic
        long_sum += 1L;

        #pragma omp atomic
        float_sum += 1.0f;

        #pragma omp atomic
        double_sum += 1.0;
    }

    EXPECT_EQ(int_sum, N);
    EXPECT_EQ(long_sum, static_cast<long>(N));
    EXPECT_FLOAT_EQ(float_sum, static_cast<float>(N));
    EXPECT_DOUBLE_EQ(double_sum, static_cast<double>(N));
}

//==============================================================================
// Test 22: Atomic update - verify mutual exclusion
//==============================================================================
TEST_F(AtomicTest, MutualExclusionGuarantee) {
    int shared_value = 0;
    std::vector<int> observed_values;
    std::atomic<bool> ready{false};
    const int N = 1000;

    // Use atomic to build a sequence
    #pragma omp parallel for ordered
    for (int i = 0; i < N; i++) {
        #pragma omp atomic
        shared_value++;
    }

    // The final value should be exactly N
    EXPECT_EQ(shared_value, N) << "Atomic should provide mutual exclusion";
}

//==============================================================================
// Test 23: Atomic operations on same location from multiple threads
//==============================================================================
TEST_F(AtomicTest, SameLocationMultipleThreads) {
    int location = 0;
    const int num_threads = omp_get_max_threads();
    const int iterations = 1000;

    #pragma omp parallel num_threads(num_threads)
    {
        for (int i = 0; i < iterations; i++) {
            #pragma omp atomic
            location++;
        }
    }

    EXPECT_EQ(location, num_threads * iterations)
        << "Multiple threads atomically updating same location should produce correct result";
}

//==============================================================================
// Test 24: Atomic with barrier synchronization
//==============================================================================
TEST_F(AtomicTest, AtomicWithBarrier) {
    int phase1_sum = 0;
    int phase2_sum = 0;
    const int N = 100;

    #pragma omp parallel num_threads(4)
    {
        // Phase 1: Each thread atomically adds values
        #pragma omp for
        for (int i = 0; i < N; i++) {
            #pragma omp atomic
            phase1_sum += i;
        }

        // Barrier ensures all phase 1 updates complete before phase 2
        #pragma omp barrier

        // Phase 2: Each thread atomically adds values again
        #pragma omp for
        for (int i = 0; i < N; i++) {
            #pragma omp atomic
            phase2_sum += i;
        }
    }

    int expected_sum = (N - 1) * N / 2;
    EXPECT_EQ(phase1_sum, expected_sum);
    EXPECT_EQ(phase2_sum, expected_sum);
}
