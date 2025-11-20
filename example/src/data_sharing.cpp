#include <emscripten.h>
#include <iostream>
#include <iomanip>

extern "C" {
    void __kmpc_fork_call(void* loc, int argc, void (*microtask)(int*, int*, ...), ...);
    int omp_get_thread_num();
    int omp_get_num_threads();
}

int main() {
    std::cout << "=== SimpleOMP Data-Sharing Clauses Demo ===\n\n";

    // Test 1: private vs shared
    std::cout << "--- Test 1: private vs shared ---\n";
    int shared_counter = 0;
    int private_var = 100;

    std::cout << "Before parallel: shared_counter=" << shared_counter
              << ", private_var=" << private_var << "\n\n";

    #pragma omp parallel num_threads(4) shared(shared_counter) private(private_var)
    {
        int tid = omp_get_thread_num();

        // Each thread modifies its private copy
        private_var = tid * 10;

        // All threads increment the shared counter
        #pragma omp atomic
        shared_counter++;

        #pragma omp critical
        {
            std::cout << "Thread " << tid << ": private_var=" << private_var
                      << " (each thread has its own copy)\n";
        }
    }

    std::cout << "\nAfter parallel: shared_counter=" << shared_counter
              << " (all threads modified it)\n";
    std::cout << "                private_var=" << private_var
              << " (unchanged, modifications were thread-local)\n\n";

    // Test 2: firstprivate
    std::cout << "--- Test 2: firstprivate ---\n";
    int initial_value = 50;

    std::cout << "Before parallel: initial_value=" << initial_value << "\n\n";

    #pragma omp parallel num_threads(4) firstprivate(initial_value)
    {
        int tid = omp_get_thread_num();

        // Each thread starts with the initial value, then modifies its copy
        int before = initial_value;
        initial_value += tid;

        #pragma omp critical
        {
            std::cout << "Thread " << tid << ": started with " << before
                      << ", modified to " << initial_value << "\n";
        }
    }

    std::cout << "\nAfter parallel: initial_value=" << initial_value
              << " (unchanged)\n\n";

    // Test 3: lastprivate
    std::cout << "--- Test 3: lastprivate ---\n";
    int result = -1;

    std::cout << "Before loop: result=" << result << "\n";
    std::cout << "Running parallel for loop (iterations 0-9)...\n\n";

    #pragma omp parallel for num_threads(4) lastprivate(result)
    for (int i = 0; i < 10; i++) {
        // Each iteration stores its index
        result = i;

        #pragma omp critical
        {
            std::cout << "  Iteration " << i << " (thread " << omp_get_thread_num()
                      << "): result=" << result << "\n";
        }
    }

    std::cout << "\nAfter loop: result=" << result
              << " (value from last iteration: 9)\n\n";

    // Test 4: Combined example
    std::cout << "--- Test 4: Combined data-sharing ---\n";
    int shared = 0;
    int fp_var = 20;
    int lp_var = 0;

    std::cout << "Computing: shared_sum += (20 + tid) for tid in 0..3\n";
    std::cout << "Before: shared=" << shared << ", firstprivate=" << fp_var << "\n\n";

    #pragma omp parallel for num_threads(4) shared(shared) firstprivate(fp_var) lastprivate(lp_var)
    for (int i = 0; i < 4; i++) {
        fp_var += i;  // Each thread: 20 + i
        lp_var = i;   // Will be 3 after loop

        #pragma omp critical
        {
            std::cout << "Iteration " << i << ": fp_var=" << fp_var
                      << ", will add to shared\n";
        }

        #pragma omp atomic
        shared += fp_var;
    }

    std::cout << "\nAfter: shared=" << shared << " (20+21+22+23=" << (20+21+22+23) << ")"
              << ", lastprivate=" << lp_var << " (from last iteration)\n";

    return 0;
}
