// Copyright 2025 Mu-Tsun Tsai
// Licensed under the MIT License

#include <gtest/gtest.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

class ReductionTest : public testing::Test {
protected:
    void SetUp() override {
        // Ensure we have multiple threads available for testing
        ASSERT_GE(omp_get_max_threads(), 2) << "Need at least 2 threads to test reduction operations";
        omp_set_num_threads(4);
    }
};

//==============================================================================
// Test 1: Basic sum reduction (+)
// Spec: Table 2.11 - Initializer: 0, Combiner: omp_out += omp_in
//==============================================================================
TEST_F(ReductionTest, BasicSumReduction) {
    const int N = 1000;
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += i;
    }

    int expected = (N - 1) * N / 2;  // Sum of 0 to 999
    EXPECT_EQ(sum, expected) << "Sum reduction should compute correct result";
}

//==============================================================================
// Test 2: Sum reduction with floating-point types
// Spec: Reduction applies to arithmetic types (int, float, double)
//==============================================================================
TEST_F(ReductionTest, FloatingPointSumReduction) {
    const int N = 1000;
    float sum_float = 0.0f;
    double sum_double = 0.0;

    #pragma omp parallel for reduction(+:sum_float, sum_double)
    for (int i = 0; i < N; i++) {
        sum_float += static_cast<float>(i);
        sum_double += static_cast<double>(i);
    }

    float expected_float = static_cast<float>((N - 1) * N / 2);
    double expected_double = static_cast<double>((N - 1) * N / 2);

    EXPECT_FLOAT_EQ(sum_float, expected_float) << "Float sum reduction should work";
    EXPECT_DOUBLE_EQ(sum_double, expected_double) << "Double sum reduction should work";
}

//==============================================================================
// Test 3: Product reduction (*)
// Spec: Table 2.11 - Initializer: 1, Combiner: omp_out *= omp_in
//==============================================================================
TEST_F(ReductionTest, ProductReduction) {
    const int N = 10;
    int product = 1;

    #pragma omp parallel for reduction(*:product)
    for (int i = 1; i <= N; i++) {
        product *= i;
    }

    EXPECT_EQ(product, 3628800) << "Product reduction should compute 10! correctly";
}

//==============================================================================
// Test 4: Subtraction reduction (-)
// Spec: Table 2.11 - Initializer: 0, Combiner: omp_out += omp_in
// Note: Despite being -, the combiner is actually += (accumulates all values)
//==============================================================================
TEST_F(ReductionTest, SubtractionReduction) {
    const int N = 100;
    int diff = 0;

    #pragma omp parallel for reduction(-:diff)
    for (int i = 0; i < N; i++) {
        diff -= i;  // Each thread subtracts its portion
    }

    // The result is the sum of all -i values
    int expected = -(N - 1) * N / 2;
    EXPECT_EQ(diff, expected) << "Subtraction reduction should accumulate all negative values";
}

//==============================================================================
// Test 5: Bitwise AND reduction (&)
// Spec: Table 2.11 - Initializer: ~0, Combiner: omp_out &= omp_in
//==============================================================================
TEST_F(ReductionTest, BitwiseANDReduction) {
    const int N = 32;
    unsigned int result = 0xFFFFFFFF;

    #pragma omp parallel for reduction(&:result)
    for (int i = 0; i < N; i++) {
        // Each iteration clears one bit
        unsigned int mask = ~(1u << i);
        result &= mask;
    }

    EXPECT_EQ(result, 0u) << "Bitwise AND reduction should clear all bits";
}

//==============================================================================
// Test 6: Bitwise OR reduction (|)
// Spec: Table 2.11 - Initializer: 0, Combiner: omp_out |= omp_in
//==============================================================================
TEST_F(ReductionTest, BitwiseORReduction) {
    const int N = 32;
    unsigned int result = 0x0;

    #pragma omp parallel for reduction(|:result)
    for (int i = 0; i < N; i++) {
        // Each iteration sets one bit
        unsigned int bit = 1u << i;
        result |= bit;
    }

    EXPECT_EQ(result, 0xFFFFFFFFu) << "Bitwise OR reduction should set all bits";
}

//==============================================================================
// Test 7: Bitwise XOR reduction (^)
// Spec: Table 2.11 - Initializer: 0, Combiner: omp_out ^= omp_in
//==============================================================================
TEST_F(ReductionTest, BitwiseXORReduction) {
    const int N = 100;
    unsigned int result = 0;

    #pragma omp parallel for reduction(^:result)
    for (int i = 0; i < N; i++) {
        // XOR each number once
        result ^= static_cast<unsigned int>(i);
    }

    // Calculate expected result sequentially
    unsigned int expected = 0;
    for (int i = 0; i < N; i++) {
        expected ^= static_cast<unsigned int>(i);
    }

    EXPECT_EQ(result, expected) << "Bitwise XOR reduction should compute correct result";
}

//==============================================================================
// Test 8: Logical AND reduction (&&)
// Spec: Table 2.11 - Initializer: 1, Combiner: omp_out = omp_in && omp_out
//==============================================================================
TEST_F(ReductionTest, LogicalANDReduction) {
    const int N = 100;

    // Test 1: All true condition
    bool all_positive = true;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i + 1;  // All positive
    }

    #pragma omp parallel for reduction(&&:all_positive)
    for (int i = 0; i < N; i++) {
        all_positive = all_positive && (data[i] > 0);
    }
    EXPECT_TRUE(all_positive) << "All values should be positive";

    // Test 2: One false condition
    data[50] = -1;
    all_positive = true;

    #pragma omp parallel for reduction(&&:all_positive)
    for (int i = 0; i < N; i++) {
        all_positive = all_positive && (data[i] > 0);
    }
    EXPECT_FALSE(all_positive) << "Should detect negative value";
}

//==============================================================================
// Test 9: Logical OR reduction (||)
// Spec: Table 2.11 - Initializer: 0, Combiner: omp_out = omp_in || omp_out
//==============================================================================
TEST_F(ReductionTest, LogicalORReduction) {
    const int N = 100;

    // Test 1: No value matches condition
    bool has_large = false;
    std::vector<int> data(N);
    for (int i = 0; i < N; i++) {
        data[i] = i;  // 0 to 99
    }

    #pragma omp parallel for reduction(||:has_large)
    for (int i = 0; i < N; i++) {
        has_large = has_large || (data[i] > 200);
    }
    EXPECT_FALSE(has_large) << "Should not find value > 200";

    // Test 2: At least one value matches
    has_large = false;
    #pragma omp parallel for reduction(||:has_large)
    for (int i = 0; i < N; i++) {
        has_large = has_large || (data[i] > 50);
    }
    EXPECT_TRUE(has_large) << "Should find value > 50";
}

//==============================================================================
// Test 10: Min reduction
// Spec: Table 2.11 - Initializer: Least representable number
//                     Combiner: omp_out = omp_in > omp_out ? omp_in : omp_out
//==============================================================================
TEST_F(ReductionTest, MinReduction) {
    const int N = 1000;
    std::vector<double> data(N);

    // Generate data with known minimum
    for (int i = 0; i < N; i++) {
        data[i] = std::sin(i * 0.1) * 100.0;
    }

    // Find the expected minimum
    double expected_min = *std::min_element(data.begin(), data.end());

    // Parallel min reduction
    double min_val = std::numeric_limits<double>::max();
    #pragma omp parallel for reduction(min:min_val)
    for (int i = 0; i < N; i++) {
        if (data[i] < min_val) {
            min_val = data[i];
        }
    }

    EXPECT_DOUBLE_EQ(min_val, expected_min) << "Min reduction should find minimum value";
}

//==============================================================================
// Test 11: Max reduction
// Spec: Table 2.11 - Initializer: Largest representable number
//                     Combiner: omp_out = omp_in < omp_out ? omp_in : omp_out
//==============================================================================
TEST_F(ReductionTest, MaxReduction) {
    const int N = 1000;
    std::vector<double> data(N);

    // Generate data with known maximum
    for (int i = 0; i < N; i++) {
        data[i] = std::sin(i * 0.1) * 100.0;
    }

    // Find the expected maximum
    double expected_max = *std::max_element(data.begin(), data.end());

    // Parallel max reduction
    double max_val = std::numeric_limits<double>::lowest();
    #pragma omp parallel for reduction(max:max_val)
    for (int i = 0; i < N; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    EXPECT_DOUBLE_EQ(max_val, expected_max) << "Max reduction should find maximum value";
}

//==============================================================================
// Test 12: Multiple reduction variables
// Spec: Any number of reduction clauses can be specified
//==============================================================================
TEST_F(ReductionTest, MultipleReductionVariables) {
    const int N = 1000;
    int sum = 0;
    int product = 1;
    int min_val = std::numeric_limits<int>::max();
    int max_val = std::numeric_limits<int>::min();

    #pragma omp parallel for reduction(+:sum) reduction(*:product) reduction(min:min_val) reduction(max:max_val)
    for (int i = 1; i <= 10; i++) {
        sum += i;
        product *= i;
        if (i < min_val) min_val = i;
        if (i > max_val) max_val = i;
    }

    EXPECT_EQ(sum, 55) << "Sum should be 1+2+...+10";
    EXPECT_EQ(product, 3628800) << "Product should be 10!";
    EXPECT_EQ(min_val, 1) << "Min should be 1";
    EXPECT_EQ(max_val, 10) << "Max should be 10";
}

//==============================================================================
// Test 13: Multiple variables in single reduction clause
// Spec: reduction clause specifies one or more list items
//==============================================================================
TEST_F(ReductionTest, MultipleVariablesSingleClause) {
    const int N = 1000;
    int sum1 = 0;
    int sum2 = 0;
    int sum3 = 0;

    #pragma omp parallel for reduction(+:sum1, sum2, sum3)
    for (int i = 0; i < N; i++) {
        sum1 += i;
        sum2 += i * 2;
        sum3 += i * 3;
    }

    int expected = (N - 1) * N / 2;
    EXPECT_EQ(sum1, expected);
    EXPECT_EQ(sum2, expected * 2);
    EXPECT_EQ(sum3, expected * 3);
}

//==============================================================================
// Test 14: Reduction with different data types
// Spec: Reduction identifier must match type for each list item
//==============================================================================
TEST_F(ReductionTest, DifferentDataTypes) {
    const int N = 1000;
    int8_t sum_i8 = 0;
    int16_t sum_i16 = 0;
    int32_t sum_i32 = 0;
    int64_t sum_i64 = 0;
    uint32_t sum_u32 = 0;

    #pragma omp parallel for reduction(+:sum_i8, sum_i16, sum_i32, sum_i64, sum_u32)
    for (int i = 0; i < 100; i++) {
        sum_i8 += 1;
        sum_i16 += 1;
        sum_i32 += 1;
        sum_i64 += 1;
        sum_u32 += 1;
    }

    EXPECT_EQ(sum_i8, static_cast<int8_t>(100));
    EXPECT_EQ(sum_i16, static_cast<int16_t>(100));
    EXPECT_EQ(sum_i32, static_cast<int32_t>(100));
    EXPECT_EQ(sum_i64, static_cast<int64_t>(100));
    EXPECT_EQ(sum_u32, static_cast<uint32_t>(100));
}

//==============================================================================
// Test 15: Reduction initializer values
// Spec: Table 2.11 specifies initializer value for each operator
//==============================================================================
TEST_F(ReductionTest, InitializerValues) {
    // Test that reduction variables are properly initialized

    // Sum: initialized to 0
    int sum = 100;  // Initial value should be overridden
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < 10; i++) {
        sum += i;
    }
    EXPECT_EQ(sum, 145) << "Sum reduction should add to original value (100 + 45)";

    // Product: initialized to 1
    int product = 2;  // Initial value should be preserved
    #pragma omp parallel for reduction(*:product)
    for (int i = 1; i <= 5; i++) {
        product *= i;
    }
    EXPECT_EQ(product, 240) << "Product reduction should multiply with original value (2 * 120)";

    // Bitwise AND: initialized to ~0
    unsigned int and_result = 0xFF00;  // Initial value should be preserved
    #pragma omp parallel for reduction(&:and_result)
    for (int i = 0; i < 8; i++) {
        and_result &= 0xFFFF;
    }
    EXPECT_EQ(and_result, 0xFF00u) << "Bitwise AND should preserve initial value";
}

//==============================================================================
// Test 16: Reduction correctness with varying thread counts
// Spec: Result should be independent of number of threads
//==============================================================================
TEST_F(ReductionTest, DifferentThreadCounts) {
    const int N = 1000;
    int expected = (N - 1) * N / 2;

    for (int num_threads = 1; num_threads <= omp_get_max_threads(); num_threads++) {
        int sum = 0;
        omp_set_num_threads(num_threads);

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < N; i++) {
            sum += i;
        }

        EXPECT_EQ(sum, expected) << "Sum should be consistent with " << num_threads << " threads";
    }
}

//==============================================================================
// Test 17: Reduction with nowait clause
// Spec: If nowait is applied, accesses to original list item create a race
//       unless synchronization ensures all threads completed
//==============================================================================
TEST_F(ReductionTest, ReductionWithNowait) {
    const int N = 1000;
    int sum = 0;

    #pragma omp parallel
    {
        #pragma omp for reduction(+:sum) nowait
        for (int i = 0; i < N; i++) {
            sum += i;
        }

        // Barrier ensures reduction completes before access
        #pragma omp barrier

        // After barrier, sum should contain the final value
        #pragma omp single
        {
            int expected = (N - 1) * N / 2;
            EXPECT_EQ(sum, expected) << "Sum should be available after barrier";
        }
    }
}

//==============================================================================
// Test 18: Reduction with complex expressions
// Spec: Reduction can involve complex computations in loop body
//==============================================================================
TEST_F(ReductionTest, ComplexExpressions) {
    const int N = 1000;
    std::vector<double> data(N);

    for (int i = 0; i < N; i++) {
        data[i] = static_cast<double>(i + 1);
    }

    double sum = 0.0;
    double sum_of_squares = 0.0;

    #pragma omp parallel for reduction(+:sum, sum_of_squares)
    for (int i = 0; i < N; i++) {
        double val = data[i];
        sum += val;
        sum_of_squares += val * val;
    }

    double mean = sum / N;
    double variance = (sum_of_squares / N) - (mean * mean);

    EXPECT_NEAR(mean, 500.5, 0.01) << "Mean should be approximately 500.5";
    EXPECT_GT(variance, 0) << "Variance should be positive";
}

//==============================================================================
// Test 19: Reduction combined with schedule clauses
// Spec: Reduction works with different schedule types
//==============================================================================
TEST_F(ReductionTest, ReductionWithSchedule) {
    const int N = 1000;
    int expected = (N - 1) * N / 2;

    // Static schedule
    int sum_static = 0;
    #pragma omp parallel for reduction(+:sum_static) schedule(static, 10)
    for (int i = 0; i < N; i++) {
        sum_static += i;
    }
    EXPECT_EQ(sum_static, expected) << "Reduction with static schedule";

    // Dynamic schedule
    int sum_dynamic = 0;
    #pragma omp parallel for reduction(+:sum_dynamic) schedule(dynamic, 10)
    for (int i = 0; i < N; i++) {
        sum_dynamic += i;
    }
    EXPECT_EQ(sum_dynamic, expected) << "Reduction with dynamic schedule";

    // Guided schedule
    int sum_guided = 0;
    #pragma omp parallel for reduction(+:sum_guided) schedule(guided, 10)
    for (int i = 0; i < N; i++) {
        sum_guided += i;
    }
    EXPECT_EQ(sum_guided, expected) << "Reduction with guided schedule";
}

//==============================================================================
// Test 20: Reduction with if clause
// Spec: Reduction should work when parallel region is active or inactive
//==============================================================================
TEST_F(ReductionTest, ReductionWithIfClause) {
    const int N = 1000;
    int expected = (N - 1) * N / 2;

    // Parallel execution
    int sum_parallel = 0;
    #pragma omp parallel for reduction(+:sum_parallel) if(true)
    for (int i = 0; i < N; i++) {
        sum_parallel += i;
    }
    EXPECT_EQ(sum_parallel, expected) << "Reduction with if(true)";

    // Serial execution
    int sum_serial = 0;
    #pragma omp parallel for reduction(+:sum_serial) if(false)
    for (int i = 0; i < N; i++) {
        sum_serial += i;
    }
    EXPECT_EQ(sum_serial, expected) << "Reduction with if(false)";
}

//==============================================================================
// Test 21: Reduction preserves original value after region
// Spec: After the end of the region, the original list item contains
//       the result of the reduction
//==============================================================================
TEST_F(ReductionTest, OriginalValuePreserved) {
    int sum = 100;  // Start with non-zero value
    const int N = 1000;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += i;
    }

    // Original variable should contain the final result
    int expected = 100 + (N - 1) * N / 2;
    EXPECT_EQ(sum, expected) << "Original variable should hold final reduction result";
}

//==============================================================================
// Test 22: Stress test - high iteration count
// Spec: Reduction should handle large workloads correctly
//==============================================================================
TEST_F(ReductionTest, HighIterationCount) {
    const int N = 1000000;
    long long sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += i;
    }

    long long expected = static_cast<long long>(N - 1) * N / 2;
    EXPECT_EQ(sum, expected) << "Large reduction should compute correctly";
}

//==============================================================================
// Test 23: Reduction with mixed operators on same variable
// Spec: A list item can appear only once in reduction clauses for that directive
// This test verifies that using different reductions on different variables works
//==============================================================================
TEST_F(ReductionTest, DifferentOperatorsOnDifferentVariables) {
    const int N = 100;
    int sum = 0;
    int product = 1;
    unsigned int bitwise_and = 0xFFFFFFFF;
    unsigned int bitwise_or = 0;

    #pragma omp parallel for reduction(+:sum) reduction(*:product) reduction(&:bitwise_and) reduction(|:bitwise_or)
    for (int i = 0; i < 10; i++) {
        sum += i;
        product *= (i + 1);
        bitwise_and &= (0xFFFFFFFF >> i);
        bitwise_or |= (1u << i);
    }

    EXPECT_EQ(sum, 45);
    EXPECT_EQ(product, 3628800);  // 10!
    EXPECT_LT(bitwise_and, 0xFFFFFFFF);
    EXPECT_EQ(bitwise_or, 0x3FFu);  // Bits 0-9 set
}

//==============================================================================
// Test 24: Reduction computation completeness
// Spec: If nowait is not specified, the reduction computation will be
//       complete at the end of the construct
//==============================================================================
TEST_F(ReductionTest, ReductionCompletenessWithoutNowait) {
    const int N = 1000;
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += i;
    }

    // At this point, reduction MUST be complete (no nowait)
    int expected = (N - 1) * N / 2;
    EXPECT_EQ(sum, expected) << "Reduction should be complete immediately after parallel region";
}

//==============================================================================
// Test 25: Floating-point reduction precision
// Spec: Reduction should maintain reasonable precision for floating-point
//==============================================================================
TEST_F(ReductionTest, FloatingPointPrecision) {
    const int N = 10000;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += 0.1;
    }

    EXPECT_NEAR(sum, N * 0.1, 0.1) << "Floating-point reduction should maintain reasonable precision";
}

//==============================================================================
// Test 26: Reduction with num_threads clause
// Spec: Reduction should work correctly with specified thread count
//==============================================================================
TEST_F(ReductionTest, ReductionWithNumThreads) {
    const int N = 1000;
    int expected = (N - 1) * N / 2;

    for (int num_threads = 1; num_threads <= 4; num_threads++) {
        int sum = 0;

        #pragma omp parallel for reduction(+:sum) num_threads(num_threads)
        for (int i = 0; i < N; i++) {
            sum += i;
        }

        EXPECT_EQ(sum, expected) << "Reduction with num_threads(" << num_threads << ")";
    }
}

//==============================================================================
// Test 27: Bitwise reductions with different patterns
// Spec: Bitwise reductions should work correctly with various bit patterns
//==============================================================================
TEST_F(ReductionTest, BitwiseReductionPatterns) {
    const int N = 8;
    unsigned int data[N] = {0xFF00, 0x0FF0, 0x00FF, 0xF0F0,
                            0x0F0F, 0xAAAA, 0x5555, 0xFFFF};

    unsigned int result_and = 0xFFFF;
    unsigned int result_or = 0x0000;
    unsigned int result_xor = 0x0000;

    #pragma omp parallel for reduction(&:result_and) reduction(|:result_or) reduction(^:result_xor)
    for (int i = 0; i < N; i++) {
        result_and &= data[i];
        result_or |= data[i];
        result_xor ^= data[i];
    }

    // Compute expected values
    unsigned int expected_and = 0xFFFF;
    unsigned int expected_or = 0x0000;
    unsigned int expected_xor = 0x0000;
    for (int i = 0; i < N; i++) {
        expected_and &= data[i];
        expected_or |= data[i];
        expected_xor ^= data[i];
    }

    EXPECT_EQ(result_and, expected_and) << "Bitwise AND with patterns";
    EXPECT_EQ(result_or, expected_or) << "Bitwise OR with patterns";
    EXPECT_EQ(result_xor, expected_xor) << "Bitwise XOR with patterns";
}

//==============================================================================
// Test 28: Reduction correctness verification - compare with sequential
// Spec: Parallel reduction result must match sequential computation
//==============================================================================
TEST_F(ReductionTest, CorrectnessVerification) {
    const int N = 1000;
    std::vector<double> data(N);

    // Generate test data
    for (int i = 0; i < N; i++) {
        data[i] = std::sin(i * 0.01) * 100.0 + std::cos(i * 0.02) * 50.0;
    }

    // Sequential computation
    double seq_sum = 0.0;
    double seq_product = 1.0;
    double seq_min = std::numeric_limits<double>::max();
    double seq_max = std::numeric_limits<double>::lowest();

    for (int i = 0; i < N; i++) {
        seq_sum += data[i];
        if (i < 10) seq_product *= (i + 1);
        if (data[i] < seq_min) seq_min = data[i];
        if (data[i] > seq_max) seq_max = data[i];
    }

    // Parallel computation
    double par_sum = 0.0;
    double par_product = 1.0;
    double par_min = std::numeric_limits<double>::max();
    double par_max = std::numeric_limits<double>::lowest();

    #pragma omp parallel for reduction(+:par_sum) reduction(*:par_product) reduction(min:par_min) reduction(max:par_max)
    for (int i = 0; i < N; i++) {
        par_sum += data[i];
        if (i < 10) par_product *= (i + 1);
        if (data[i] < par_min) par_min = data[i];
        if (data[i] > par_max) par_max = data[i];
    }

    EXPECT_NEAR(par_sum, seq_sum, 0.01) << "Parallel sum should match sequential";
    EXPECT_DOUBLE_EQ(par_product, seq_product) << "Parallel product should match sequential";
    EXPECT_DOUBLE_EQ(par_min, seq_min) << "Parallel min should match sequential";
    EXPECT_DOUBLE_EQ(par_max, seq_max) << "Parallel max should match sequential";
}
