#include <iostream>
#include <chrono>

int main() {
    int n = 100;

    // Test 1: n=100, should execute serially
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for if(n > 1000) num_threads(8)
    for (int i = 0; i < 100000000; i++) {
        volatile int x = i * i;
    }
    auto end = std::chrono::high_resolution_clock::now();
	auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "n=100 (serial): "
              << time << " ms (" << time / 100.0 << "ms per iteration)" << std::endl;

    // Test 2: n=5000, should execute in parallel
    n = 5000;
    start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for if(n > 1000) num_threads(8)
    for (int i = 0; i < 100000000; i++) {
        volatile int x = i * i;
    }
    end = std::chrono::high_resolution_clock::now();
	time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "n=5000 (parallel): "
              << time << " ms (" << time / 5000.0 << "ms per iteration)" << std::endl;

    return 0;
}