// SimpleOMP Example: Nowait Clause
// Demonstrates the nowait clause to skip implicit barriers

#include <iostream>
#include <chrono>
#include <thread>
#include <omp.h>

// Helper function to simulate work
void simulate_work(int ms) {
	std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
	const int num_threads = 4;

	std::cout << "=== SimpleOMP nowait Clause Demonstration ===" << std::endl;
	std::cout << "This demo shows the difference between constructs with and "
				 "without nowait."
			  << std::endl;
	std::cout << std::endl;

	// Test 1: Single without nowait (has implicit barrier)
	std::cout << "--- Test 1: Single WITHOUT nowait (implicit barrier) ---"
			  << std::endl;
	auto start1 = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

#pragma omp single
		{
#pragma omp critical
			std::cout << "Thread " << tid
					  << " executing single block (doing 200ms work)"
					  << std::endl;
			simulate_work(200);
		}
		// Implicit barrier here - all threads wait

#pragma omp critical
		std::cout << "Thread " << tid << " continues after single"
				  << std::endl;
	}

	auto end1 = std::chrono::high_resolution_clock::now();
	auto duration1 =
		std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1)
			.count();

	std::cout << "Time: " << duration1
			  << "ms (all threads waited at implicit barrier)" << std::endl;
	std::cout << std::endl;

	// Test 2: Single with nowait (no barrier)
	std::cout << "--- Test 2: Single WITH nowait (no barrier) ---" << std::endl;
	auto start2 = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

#pragma omp single nowait
		{
#pragma omp critical
			std::cout << "Thread " << tid
					  << " executing single block (doing 200ms work)"
					  << std::endl;
			simulate_work(200);
		}
		// NO barrier - other threads continue immediately

#pragma omp critical
		std::cout << "Thread " << tid << " continues (without waiting)"
				  << std::endl;
	}

	auto end2 = std::chrono::high_resolution_clock::now();
	auto duration2 =
		std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2)
			.count();

	std::cout << "Time: " << duration2 << "ms (other threads didn't wait)"
			  << std::endl;
	std::cout << std::endl;

	// Test 3: For loop without nowait
	std::cout << "--- Test 3: For loop WITHOUT nowait (implicit barrier) ---"
			  << std::endl;
	int counter1 = 0;

	auto start3 = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

#pragma omp for
		for(int i = 0; i < 4; i++) {
#pragma omp critical
			std::cout << "Thread " << tid << " processing iteration " << i
					  << std::endl;
			simulate_work(50);
		}
		// Implicit barrier here

#pragma omp critical
		{
			counter1++;
			std::cout << "Thread " << tid
					  << " reached after-loop section (counter=" << counter1
					  << ")" << std::endl;
		}
	}

	auto end3 = std::chrono::high_resolution_clock::now();
	auto duration3 =
		std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3)
			.count();

	std::cout << "Time: " << duration3 << "ms" << std::endl;
	std::cout << std::endl;

	// Test 4: For loop with nowait
	std::cout << "--- Test 4: For loop WITH nowait (no barrier) ---"
			  << std::endl;
	int counter2 = 0;

	auto start4 = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(num_threads)
	{
		int tid = omp_get_thread_num();

#pragma omp for nowait
		for(int i = 0; i < 4; i++) {
#pragma omp critical
			std::cout << "Thread " << tid << " processing iteration " << i
					  << std::endl;
			simulate_work(50);
		}
		// NO barrier - threads that finish early can continue

#pragma omp critical
		{
			counter2++;
			std::cout << "Thread " << tid
					  << " reached after-loop section (counter=" << counter2
					  << ")" << std::endl;
		}
	}

	auto end4 = std::chrono::high_resolution_clock::now();
	auto duration4 =
		std::chrono::duration_cast<std::chrono::milliseconds>(end4 - start4)
			.count();

	std::cout << "Time: " << duration4 << "ms" << std::endl;
	std::cout << std::endl;

	// Summary
	std::cout << "=== Summary ===" << std::endl;
	std::cout << "Without nowait: All threads synchronize at implicit barrier"
			  << std::endl;
	std::cout << "With nowait: Threads continue immediately without waiting"
			  << std::endl;
	std::cout << std::endl;
	std::cout << "Test 1 (single, no nowait): " << duration1 << "ms"
			  << std::endl;
	std::cout << "Test 2 (single, nowait): " << duration2
			  << "ms (should be faster)" << std::endl;
	std::cout << "Test 3 (for, no nowait): " << duration3 << "ms" << std::endl;
	std::cout << "Test 4 (for, nowait): " << duration4
			  << "ms (threads may finish at different times)" << std::endl;

	std::cout << "\n✓ nowait clause demonstration completed!" << std::endl;

	return 0;
}
