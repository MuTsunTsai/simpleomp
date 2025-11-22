// Copyright 2025 Mu-Tsun Tsai
// Licensed under the MIT License

#include <gtest/gtest.h>
#include <omp.h>
#include <atomic>
#include <vector>

class IfClauseTest : public testing::Test {
protected:
    void SetUp() override {
        // Ensure we have multiple threads available for testing
        ASSERT_GE(omp_get_max_threads(), 2) << "Need at least 2 threads to test if clause behavior";
    }
};

// Test 1: Basic if clause - condition evaluates to false
// Expected: Creates inactive parallel region (single thread execution)
TEST_F(IfClauseTest, FalseConditionCreatesInactiveRegion) {
    std::atomic<int> thread_count{0};
    int max_thread_num = -1;

    #pragma omp parallel if(false)
    {
        thread_count++;
        int tid = omp_get_thread_num();
        if (tid > max_thread_num) max_thread_num = tid;
    }

    // According to OpenMP spec: if(false) creates an "inactive parallel region"
    // with only one thread (thread 0)
    EXPECT_EQ(thread_count.load(), 1) << "if(false) should execute with single thread";
    EXPECT_EQ(max_thread_num, 0) << "Only thread 0 should execute when if(false)";
}

// Test 2: Basic if clause - condition evaluates to true
// Expected: Creates active parallel region (multiple threads)
TEST_F(IfClauseTest, TrueConditionCreatesActiveRegion) {
    std::atomic<int> thread_count{0};
    omp_set_num_threads(4);

    #pragma omp parallel if(true)
    {
        thread_count++;
    }

    // if(true) should create an active parallel region with multiple threads
    EXPECT_GT(thread_count.load(), 1) << "if(true) should execute with multiple threads";
}

// Test 3: if clause with runtime expression - parallel for construct
// Expected: Behavior switches based on runtime condition
TEST_F(IfClauseTest, RuntimeConditionInParallelFor) {
    const int small_n = 10;
    const int large_n = 10000;

    // Test with small_n (should be serial)
    std::atomic<int> thread_count_small{0};
    #pragma omp parallel for if(small_n > 1000)
    for (int i = 0; i < small_n; i++) {
        thread_count_small++;
    }

    // Test with large_n (should be parallel)
    std::atomic<int> unique_threads{0};
    std::vector<bool> thread_participated(omp_get_max_threads(), false);

    #pragma omp parallel for if(large_n > 1000)
    for (int i = 0; i < large_n; i++) {
        int tid = omp_get_thread_num();
        if (!thread_participated[tid]) {
            thread_participated[tid] = true;
            unique_threads++;
        }
    }

    EXPECT_EQ(thread_count_small.load(), small_n) << "Small loop should complete all iterations";
    EXPECT_GT(unique_threads.load(), 1) << "Large loop with if(true) should use multiple threads";
}

// Test 4: if clause interaction with num_threads
// Expected: if(false) overrides num_threads, using only 1 thread
TEST_F(IfClauseTest, IfClauseTakesPrecedenceOverNumThreads) {
    std::atomic<int> thread_count{0};

    #pragma omp parallel num_threads(4) if(false)
    {
        thread_count++;
    }

    // if(false) should override num_threads(4) and use only 1 thread
    EXPECT_EQ(thread_count.load(), 1)
        << "if(false) should override num_threads and execute with single thread";
}

// Test 5: if clause with complex expression
// Expected: Expression is evaluated once before parallel region
TEST_F(IfClauseTest, ComplexExpressionEvaluation) {
    int evaluation_count = 0;
    auto condition = [&evaluation_count]() -> bool {
        evaluation_count++;
        return true;
    };

    #pragma omp parallel if(condition())
    {
        // Empty parallel region
    }

    // The if clause expression should be evaluated exactly once
    EXPECT_EQ(evaluation_count, 1)
        << "if clause expression should be evaluated exactly once";
}

// Test 6: if clause in nested parallel regions
// Expected: Each parallel region's if clause is evaluated independently
TEST_F(IfClauseTest, NestedParallelRegionsWithIfClause) {
    std::atomic<int> outer_threads{0};
    std::atomic<int> inner_threads{0};

    #pragma omp parallel if(true) num_threads(2)
    {
        outer_threads++;

        // Inner parallel region with if(false) - should be inactive
        #pragma omp parallel if(false) num_threads(4)
        {
            inner_threads++;
        }
    }

    EXPECT_GE(outer_threads.load(), 2) << "Outer region should use multiple threads";
    EXPECT_EQ(inner_threads.load(), outer_threads.load())
        << "Each outer thread should spawn exactly 1 inner thread (due to if(false))";
}

// Test 7: if clause with variable condition
// Expected: Same behavior whether condition is literal or variable
TEST_F(IfClauseTest, VariableVsLiteralCondition) {
    std::atomic<int> count_with_literal{0};
    std::atomic<int> count_with_variable{0};
    bool condition = false;

    #pragma omp parallel if(false)
    {
        count_with_literal++;
    }

    #pragma omp parallel if(condition)
    {
        count_with_variable++;
    }

    EXPECT_EQ(count_with_literal.load(), 1);
    EXPECT_EQ(count_with_variable.load(), 1);
    EXPECT_EQ(count_with_literal.load(), count_with_variable.load())
        << "Literal false and variable false should behave identically";
}

// Test 8: if clause correctness - work distribution
// Expected: Results should be identical regardless of if clause value
TEST_F(IfClauseTest, ResultCorrectnessWithIfClause) {
    const int N = 1000;
    std::vector<int> result_serial(N, 0);
    std::vector<int> result_parallel(N, 0);

    // Serial execution via if(false)
    #pragma omp parallel for if(false)
    for (int i = 0; i < N; i++) {
        result_serial[i] = i * i;
    }

    // Parallel execution via if(true)
    #pragma omp parallel for if(true)
    for (int i = 0; i < N; i++) {
        result_parallel[i] = i * i;
    }

    // Results should be identical
    EXPECT_EQ(result_serial, result_parallel)
        << "Computation results should be identical whether executed serially or in parallel";
}

// Test 9: if clause with omp_get_num_threads() and omp_get_thread_num()
// Expected: Behavior consistent with inactive/active region definitions
TEST_F(IfClauseTest, ThreadQueryFunctionsWithIfClause) {
    int num_threads_inactive = -1;
    int num_threads_active = -1;

    #pragma omp parallel if(false)
    {
        #pragma omp master
        {
            num_threads_inactive = omp_get_num_threads();
        }
    }

    #pragma omp parallel if(true) num_threads(4)
    {
        #pragma omp master
        {
            num_threads_active = omp_get_num_threads();
        }
    }

    EXPECT_EQ(num_threads_inactive, 1)
        << "Inactive parallel region (if(false)) should report 1 thread";
    EXPECT_GT(num_threads_active, 1)
        << "Active parallel region (if(true)) should report multiple threads";
}

// Test 10: if clause combined with schedule clause
// Expected: Schedule clause should still apply even with if(false)
TEST_F(IfClauseTest, IfClauseWithScheduleClause) {
    const int N = 100;
    std::vector<int> result_static(N, 0);
    std::vector<int> result_dynamic(N, 0);

    // Even with if(false), schedule clause syntax should be accepted
    #pragma omp parallel for if(false) schedule(static, 10)
    for (int i = 0; i < N; i++) {
        result_static[i] = i * 2;
    }

    #pragma omp parallel for if(false) schedule(dynamic, 5)
    for (int i = 0; i < N; i++) {
        result_dynamic[i] = i * 2;
    }

    // Both should produce correct results (though executed serially)
    for (int i = 0; i < N; i++) {
        EXPECT_EQ(result_static[i], i * 2);
        EXPECT_EQ(result_dynamic[i], i * 2);
    }
}
