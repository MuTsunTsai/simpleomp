#include <iostream>

using namespace std;

// External OpenMP functions
extern "C" {
	int omp_get_thread_num();
}

int main() {
	cout << "=== SimpleOMP Barrier Construct Demo ===" << endl << endl;

	int phase1_complete = 0;
	int phase2_complete = 0;

#pragma omp parallel num_threads(4)
	{
		int tid = omp_get_thread_num();

		// Phase 1: All threads perform initial work
#pragma omp critical
		{
			cout << "Thread " << tid << ": Entering Phase 1" << endl;
			phase1_complete++;
		}

		// Barrier: Wait for all threads to complete Phase 1
#pragma omp barrier

		// Phase 2: All threads can safely read phase1_complete
		// because barrier ensures all threads finished Phase 1
#pragma omp critical
		{
			cout << "Thread " << tid
				 << ": Phase 1 complete (count=" << phase1_complete
				 << "), entering Phase 2" << endl;
			phase2_complete++;
		}

		// Another barrier before final check
#pragma omp barrier

		// Phase 3: Final synchronization check
#pragma omp critical
		{
			cout << "Thread " << tid
				 << ": Phase 2 complete (count=" << phase2_complete << ")"
				 << endl;
		}
	}

	cout << endl << "All phases completed successfully!" << endl;
	cout << "Phase 1 count: " << phase1_complete << " (expected: 4)" << endl;
	cout << "Phase 2 count: " << phase2_complete << " (expected: 4)" << endl;

	return 0;
}
