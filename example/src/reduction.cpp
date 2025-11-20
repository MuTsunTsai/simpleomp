// OpenMP Reduction Example
// Demonstrates #pragma omp reduction clause for parallel aggregation

#include <iostream>
#include <vector>
#include <cmath>

// Forward declarations for OpenMP runtime functions
extern "C" {
    void omp_set_num_threads(int num_threads);
    int omp_get_num_threads();
    int omp_get_thread_num();
    int omp_get_max_threads();
}

// Example 1: Sum reduction (most common use case)
void example_sum_reduction() {
    std::cout << "\n=== Example 1: Sum Reduction ===\n";

    const int N = 1000;
    std::vector<int> data(N);

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = i + 1;  // 1, 2, 3, ..., 1000
    }

    int sum = 0;

    // Parallel sum reduction
    // Each thread computes partial sum, then all partial sums are combined
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += data[i];
    }

    // Expected: sum of 1 to 1000 = 1000 * 1001 / 2 = 500500
    std::cout << "Sum of 1 to " << N << " = " << sum << "\n";
    std::cout << "Expected: " << (N * (N + 1) / 2) << "\n";
}

// Example 2: Product reduction
void example_product_reduction() {
    std::cout << "\n=== Example 2: Product Reduction ===\n";

    const int N = 10;
    std::vector<int> data(N);

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = i + 1;  // 1, 2, 3, ..., 10
    }

    int product = 1;

    // Parallel product reduction
    #pragma omp parallel for reduction(*:product)
    for (int i = 0; i < N; i++) {
        product *= data[i];
    }

    // Expected: 10! = 3628800
    std::cout << "Product of 1 to " << N << " (factorial) = " << product << "\n";
    std::cout << "Expected: 3628800\n";
}

// Example 3: Min/Max reduction
void example_min_max_reduction() {
    std::cout << "\n=== Example 3: Min/Max Reduction ===\n";

    const int N = 1000;
    std::vector<double> data(N);

    // Initialize with some pattern
    for (int i = 0; i < N; i++) {
        data[i] = std::sin(i * 0.1) * 100.0;
    }

    double min_val = data[0];
    double max_val = data[0];

    // Parallel min/max reduction
    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < N; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }

    std::cout << "Min value: " << min_val << "\n";
    std::cout << "Max value: " << max_val << "\n";
}

// Example 4: Multiple variables reduction
void example_multiple_reduction() {
    std::cout << "\n=== Example 4: Multiple Variables Reduction ===\n";

    const int N = 1000;
    std::vector<double> data(N);

    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = static_cast<double>(i + 1);
    }

    double sum = 0.0;
    double sum_of_squares = 0.0;
    int count = 0;

    // Reduce multiple variables simultaneously
    #pragma omp parallel for reduction(+:sum,sum_of_squares,count)
    for (int i = 0; i < N; i++) {
        sum += data[i];
        sum_of_squares += data[i] * data[i];
        count++;
    }

    double mean = sum / count;
    double variance = (sum_of_squares / count) - (mean * mean);
    double stddev = std::sqrt(variance);

    std::cout << "Count: " << count << "\n";
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Standard Deviation: " << stddev << "\n";
}

// Example 5: Logical AND/OR reduction
void example_logical_reduction() {
    std::cout << "\n=== Example 5: Logical AND/OR Reduction ===\n";

    const int N = 100;
    std::vector<int> data(N);

    // All positive numbers
    for (int i = 0; i < N; i++) {
        data[i] = i + 1;
    }

    bool all_positive = true;
    bool any_large = false;

    #pragma omp parallel for reduction(&&:all_positive) reduction(||:any_large)
    for (int i = 0; i < N; i++) {
        all_positive = all_positive && (data[i] > 0);
        any_large = any_large || (data[i] > 50);
    }

    std::cout << "All values positive: " << (all_positive ? "Yes" : "No") << "\n";
    std::cout << "Any value > 50: " << (any_large ? "Yes" : "No") << "\n";
}

// Example 6: Bitwise reduction
void example_bitwise_reduction() {
    std::cout << "\n=== Example 6: Bitwise Reduction ===\n";

    const int N = 8;
    unsigned int data[N] = {0xFF00, 0x0FF0, 0x00FF, 0xF0F0,
                            0x0F0F, 0xAAAA, 0x5555, 0xFFFF};

    unsigned int bitwise_and = 0xFFFF;
    unsigned int bitwise_or = 0x0000;
    unsigned int bitwise_xor = 0x0000;

    #pragma omp parallel for reduction(&:bitwise_and) reduction(|:bitwise_or) reduction(^:bitwise_xor)
    for (int i = 0; i < N; i++) {
        bitwise_and &= data[i];
        bitwise_or |= data[i];
        bitwise_xor ^= data[i];
    }

    std::cout << "Bitwise AND: 0x" << std::hex << bitwise_and << std::dec << "\n";
    std::cout << "Bitwise OR:  0x" << std::hex << bitwise_or << std::dec << "\n";
    std::cout << "Bitwise XOR: 0x" << std::hex << bitwise_xor << std::dec << "\n";
}

int main() {
    std::cout << "OpenMP Reduction Operations Demo\n";
    std::cout << "=================================\n";

    int max_threads = omp_get_max_threads();
    std::cout << "Max threads available: " << max_threads << "\n";

    // Set number of threads
    omp_set_num_threads(4);

    // Run all examples
    example_sum_reduction();
    example_product_reduction();
    example_min_max_reduction();
    example_multiple_reduction();
    example_logical_reduction();
    example_bitwise_reduction();

    std::cout << "\n=== All Examples Completed ===\n";

    return 0;
}
