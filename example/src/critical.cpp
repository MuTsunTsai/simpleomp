#include <iostream>

using namespace std;

// External OpenMP functions
extern "C" {
	int omp_get_thread_num();
}

int main() {
	cout << "=== SimpleOMP Critical Construct Demo ===" << endl << endl;

	int shared_counter = 0;

#pragma omp parallel num_threads(4)
	{
		int tid = omp_get_thread_num();

		// Without critical: race condition
		for(int i = 0; i < 1000; i++) {
#pragma omp critical
			{
				// Only one thread can execute this at a time
				shared_counter++;
			}
		}

#pragma omp critical
		{
			cout << "Thread " << tid << " finished incrementing" << endl;
		}
	}

	cout << endl
		 << "Final counter value: " << shared_counter << " (expected: 4000)"
		 << endl;

	return 0;
}
