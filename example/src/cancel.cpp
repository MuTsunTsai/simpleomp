// SimpleOMP Example: Cancellation
// Demonstrates #pragma omp cancel and #pragma omp cancellation point

#include <iostream>
#include <omp.h>

int main() {
	std::cout << "=== SimpleOMP Cancellation Demo ===" << std::endl;
	std::cout << std::endl;

	// Check if cancellation is enabled
	if(omp_get_cancellation()) {
		std::cout << "✓ Cancellation is ENABLED" << std::endl;
	} else {
		std::cout << "✗ Cancellation is DISABLED" << std::endl;
		std::cout << "  Set OMP_CANCELLATION=true to enable cancellation"
				  << std::endl;
		return 1;
	}

	std::cout << std::endl;

	// Test 1: Simple parallel region with cancellation
	{
		std::cout << "=== Test 1: Parallel Region with Cancel ===" << std::endl;
		std::cout
			<< "Thread 0 will request cancellation, others should detect it\n"
			<< std::endl;

		// Track how many iterations each thread completed before cancellation
		int iter[4] = {0, 0, 0, 0};

#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();

#pragma omp critical
			{
				std::cout << "Thread " << tid << " started" << std::endl;
			}

			// Use barrier to ensure all threads start together
#pragma omp barrier

			// Thread 0 requests cancellation immediately (no delay)
			if(tid == 0) {
#pragma omp critical
				{
					std::cout << "Thread 0 requesting cancellation..."
							  << std::endl;
				}

#pragma omp cancel parallel
			}

			// Other threads perform work and check for cancellation periodically
			// Using sqrt() to prevent compiler from optimizing away the loop
			double delay = 0.0;
			if(tid != 0) {
				for(int i = 0; i < 5000; i++) {
					delay = sqrt(delay + i);
					iter[tid] = i;  // Record progress

					// Check if cancellation was requested
					// If cancelled, thread jumps to end of parallel region
#pragma omp cancellation point parallel
				}
			}

			// Threads that complete normally (not cancelled) will print this
			// Cancelled threads skip this and jump directly to end of parallel region
#pragma omp critical
			{
				std::cout << "Thread " << tid
						  << " completed normally with result " << delay
						  << std::endl;
			}
		}

		// Check which threads were cancelled by examining their iteration count
		// If a thread completed all 5000 iterations, iter[i] would be 4999
		// If iter[i] < 4999, the thread was cancelled mid-execution
		for(int i = 1; i < 4; i++) {
			if(iter[i] < 4999) {
				std::cout << "Thread " << i << " was cancelled at iteration "
						  << iter[i] << std::endl;
			}
		}

		std::cout << "\n✓ Test 1 complete\n" << std::endl;
	}

	// Test 2: Cancellation in a loop
	{
		std::cout << "=== Test 2: Loop with Cancellation ===" << std::endl;
		std::cout << "Stop processing when error is found (iteration 50)\n"
				  << std::endl;

		// Shared variables to track error state and work completed
		int error_found = 0;
		int processed_count = 0;

#pragma omp parallel for num_threads(4) shared(error_found, processed_count)
		for(int i = 0; i < 100; i++) {
			int tid = omp_get_thread_num();

			// Simulate finding an error at iteration 50
			if(i == 50) {
#pragma omp critical
				{
					std::cout << "Thread " << tid
							  << " found error at iteration " << i << std::endl;
					error_found = 1;
				}

#pragma omp cancel for
			}

			// Check for cancellation before processing this iteration
			// If cancelled, thread exits the loop immediately
#pragma omp cancellation point for

			// Only count iterations that weren't cancelled
			if(!error_found) {
#pragma omp atomic
				processed_count++;
			}
		}

		std::cout << "\nProcessed " << processed_count
				  << " iterations before cancellation" << std::endl;
		std::cout << "(Note: Due to parallel execution, some iterations after "
					 "50 may still execute)"
				  << std::endl;
		std::cout << "\n✓ Test 2 complete\n" << std::endl;
	}

	std::cout << "=== All Cancellation Tests Complete ===" << std::endl;
	std::cout
		<< "\nNote: SimpleOMP implements cancellation via __kmpc_cancel() and"
		<< std::endl;
	std::cout << "__kmpc_cancellationpoint() runtime functions. The compiler "
				 "generates"
			  << std::endl;
	std::cout << "appropriate control flow to jump to the end of regions when "
				 "cancelled."
			  << std::endl;

	return 0;
}
