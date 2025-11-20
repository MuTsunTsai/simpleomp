#include <iostream>
#include <omp.h>

using namespace std;

int main() {
	cout << "=== SimpleOMP Single Construct Demo ===" << endl << endl;

	int initialization_value = 0;
	int *shared_data = nullptr;

#pragma omp parallel num_threads(4)
	{
		int tid = omp_get_thread_num();

		// All threads execute this
#pragma omp critical
		{
			cout << "Thread " << tid << ": Before single block" << endl;
		}

#pragma omp single
		{
			// Only ONE thread (whichever arrives first) executes this
			initialization_value = 42;
			shared_data = new int[10];
			for(int i = 0; i < 10; i++) {
				shared_data[i] = i * 10;
			}
			cout << ">>> Thread " << tid
				 << ": Inside single block (initializing data) <<<" << endl;
		}

		// Implicit barrier at end of single construct ensures
		// all threads see the initialized data

		// All threads execute this
#pragma omp critical
		{
			cout << "Thread " << tid
				 << ": After single block (initialization_value="
				 << initialization_value << ")" << endl;
		}

		// Demonstrate another single construct
#pragma omp single
		{
			cout << ">>> Thread " << tid << ": Inside second single block <<<"
				 << endl;
		}

		// All threads execute this
#pragma omp critical
		{
			cout << "Thread " << tid << ": After second single block" << endl;
		}
	}

	cout << endl << "Initialization completed successfully!" << endl;
	cout << "Final initialization_value: " << initialization_value << endl;
	cout << "Shared data[5]: " << shared_data[5] << " (expected: 50)" << endl;

	delete[] shared_data;

	return 0;
}
