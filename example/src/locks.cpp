// SimpleOMP Example: OpenMP Locks
// Demonstrates omp_lock_t and omp_nest_lock_t for thread synchronization

#include <iostream>
#include <omp.h>

int main() {
	std::cout << "=== SimpleOMP Locks Demo ===" << std::endl;
	std::cout << std::endl;

	// Test 1: Simple Lock (omp_lock_t)
	{
		std::cout << "Test 1: Simple Lock (omp_lock_t)" << std::endl;
		std::cout << "4 threads incrementing a shared counter using locks"
				  << std::endl;

		omp_lock_t lock;
		omp_init_lock(&lock);

		int counter = 0;

#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();

			for(int i = 0; i < 1000; i++) {
				omp_set_lock(&lock);
				// Critical section: only one thread can execute at a time
				counter++;
				omp_unset_lock(&lock);
			}

#pragma omp critical
			{
				std::cout << "Thread " << tid << " finished" << std::endl;
			}
		}

		omp_destroy_lock(&lock);

		std::cout << "Expected: 4000, Actual: " << counter << std::endl;
		std::cout << (counter == 4000 ? "✓ PASS" : "✗ FAIL") << std::endl;
		std::cout << std::endl;
	}

	// Test 2: Test Lock (non-blocking)
	{
		std::cout << "Test 2: Test Lock (omp_test_lock)" << std::endl;
		std::cout << "Demonstrates non-blocking lock acquisition" << std::endl;

		omp_lock_t lock;
		omp_init_lock(&lock);

		int successful_locks = 0;
		int failed_attempts = 0;

#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();
			int my_successes = 0;
			int my_failures = 0;

			for(int i = 0; i < 100; i++) {
				if(omp_test_lock(&lock)) {
					// Successfully acquired the lock
					my_successes++;

					// Simulate some work
					volatile int dummy = 0;
					for(int j = 0; j < 1000; j++) {
						dummy = j;
					}

					omp_unset_lock(&lock);
				} else {
					// Failed to acquire lock (another thread holds it)
					my_failures++;
				}
			}

#pragma omp critical
			{
				successful_locks += my_successes;
				failed_attempts += my_failures;
				std::cout << "Thread " << tid << ": " << my_successes
						  << " successes, " << my_failures << " failures"
						  << std::endl;
			}
		}

		omp_destroy_lock(&lock);

		std::cout << "Total attempts: " << (successful_locks + failed_attempts)
				  << " (400 expected)" << std::endl;
		std::cout << "Successful locks: " << successful_locks << std::endl;
		std::cout << "Failed attempts: " << failed_attempts
				  << " (contention occurred)" << std::endl;
		std::cout << std::endl;
	}

	// Test 3: Nestable Lock (omp_nest_lock_t)
	{
		std::cout << "Test 3: Nestable Lock (omp_nest_lock_t)" << std::endl;
		std::cout << "Demonstrates locks that can be acquired multiple times by "
					 "the same thread"
				  << std::endl;

		omp_nest_lock_t nest_lock;
		omp_init_nest_lock(&nest_lock);

		int counter = 0;

#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();

			for(int i = 0; i < 100; i++) {
				// Acquire lock first time
				omp_set_nest_lock(&nest_lock);

				// Acquire lock second time (nesting!)
				omp_set_nest_lock(&nest_lock);

				// Acquire lock third time
				omp_set_nest_lock(&nest_lock);

				// Critical section with triple nesting
				counter++;

				// Release locks in reverse order
				omp_unset_nest_lock(&nest_lock);
				omp_unset_nest_lock(&nest_lock);
				omp_unset_nest_lock(&nest_lock);
			}

#pragma omp critical
			{
				std::cout << "Thread " << tid << " completed nested locking"
						  << std::endl;
			}
		}

		omp_destroy_nest_lock(&nest_lock);

		std::cout << "Expected: 400, Actual: " << counter << std::endl;
		std::cout << (counter == 400 ? "✓ PASS" : "✗ FAIL") << std::endl;
		std::cout << std::endl;
	}

	// Test 4: Test Nestable Lock
	{
		std::cout << "Test 4: Test Nestable Lock (omp_test_nest_lock)"
				  << std::endl;
		std::cout << "Demonstrates non-blocking nestable lock acquisition"
				  << std::endl;

		omp_nest_lock_t nest_lock;
		omp_init_nest_lock(&nest_lock);

#pragma omp parallel num_threads(4)
		{
			int tid = omp_get_thread_num();

			// Try to acquire the lock
			int nesting_level = omp_test_nest_lock(&nest_lock);

			if(nesting_level > 0) {
				// Successfully acquired (nesting_level = 1)

				// Try to acquire again (same thread)
				int level2 = omp_test_nest_lock(&nest_lock);

#pragma omp critical
				{
					std::cout << "Thread " << tid << ": First level = "
							  << nesting_level << ", Second level = " << level2
							  << std::endl;
				}

				// Release twice
				omp_unset_nest_lock(&nest_lock);
				omp_unset_nest_lock(&nest_lock);
			} else {
#pragma omp critical
				{
					std::cout << "Thread " << tid
							  << ": Failed to acquire lock" << std::endl;
				}
			}
		}

		omp_destroy_nest_lock(&nest_lock);

		std::cout << "✓ Nestable lock test completed" << std::endl;
		std::cout << std::endl;
	}

	std::cout << "=== All Lock Tests Complete ===" << std::endl;

	return 0;
}
