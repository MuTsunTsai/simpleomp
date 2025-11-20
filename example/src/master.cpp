#include <iostream>
#include <omp.h>

using namespace std;

int main() {
	cout << "=== SimpleOMP Master Construct Demo ===" << endl << endl;

	int counter = 0;

#pragma omp parallel num_threads(4)
	{
		int tid = omp_get_thread_num();

		// All threads execute this
#pragma omp critical
		{
			cout << "Thread " << tid << ": Before master block" << endl;
		}

#pragma omp master
		{
			// Only master thread (thread 0) executes this
			counter = 100;
			cout << ">>> Thread " << tid
				 << ": Inside master block (setting counter = 100) <<<" << endl;
		}

		// All threads execute this
#pragma omp critical
		{
			cout << "Thread " << tid << ": After master block" << endl;
		}
	}

	cout << endl << "Final counter value: " << counter << endl;

	return 0;
}
