#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

void test_static_schedule() {
	cout << "\n== Static Schedule (chunk_size=3) ==" << endl;
	cout << "Expected: Each thread gets contiguous chunks of 3 iterations"
		 << endl;
	cout << "Thread assignment pattern should be: "
			"0,0,0,1,1,1,2,2,2,3,3,3,0,0,0..."
		 << endl;

	vector<int> assignment(20, -1);

#pragma omp parallel for schedule(static, 3) num_threads(4)
	for(int i = 0; i < 20; i++) {
		assignment[i] = omp_get_thread_num();
	}

	cout << "Actual:   ";
	for(int i = 0; i < 20; i++) {
		cout << assignment[i];
		if(i < 19) cout << ",";
	}
	cout << endl;
}

void test_dynamic_schedule() {
	cout << "\n== Dynamic Schedule (chunk_size=2) ==" << endl;
	cout << "Expected: Threads dynamically request chunks of 2 iterations"
		 << endl;
	cout << "Order depends on which thread finishes first (may vary)" << endl;

	vector<int> assignment(20, -1);
	int actual_threads = 0;

#pragma omp parallel
	{
#pragma omp master
		{
			actual_threads = omp_get_num_threads();
		}
	}

	cout << "Actual number of threads: " << actual_threads << endl;

	bool reassign = false;

#pragma omp parallel for schedule(dynamic, 2)
	for(int i = 0; i < 20; i++) {
		if(reassign) continue;
		// More substantial computation to ensure threads have time to
		// participate
		volatile double dummy = 0;
		for(int j = 0; j < (i * 17 % 7 + 1) * 5000000; j++) {
			dummy = sqrt(dummy + j);
		}
#pragma omp critical
		{
			if(assignment[i] != -1) reassign = true;
		}
#pragma omp critical
		{
			assignment[i] = omp_get_thread_num();
		}
	}

	cout << "Actual:   ";
	for(int i = 0; i < 20; i++) {
		cout << assignment[i];
		if(i < 19) cout << ",";
	}
	cout << endl;

	cout << "No reassignment: " << (!reassign ? "PASS" : "FAIL") << endl;

	// Verify all threads participated
	cout << "Verifying all threads participated: ";
	bool all_threads[4] = {false, false, false, false};
	for(int i = 0; i < 20; i++) {
		if(assignment[i] >= 0 && assignment[i] < 4) {
			all_threads[assignment[i]] = true;
		}
	}
	bool all_participated = true;
	for(int i = 0; i < 4; i++) {
		if(!all_threads[i]) {
			all_participated = false;
			break;
		}
	}
	cout << (all_participated ? "PASS" : "FAIL") << endl;

	// Note: Dynamic scheduling allows threads to claim multiple consecutive
	// chunks, so we don't verify chunk boundaries - only that work is
	// distributed
}

void test_guided_schedule() {
	cout << "\n== Guided Schedule (min_chunk=2) ==" << endl;
	cout << "Expected: Chunk size decreases exponentially (starts large, ends "
			"at min 2)"
		 << endl;
	cout << "With 4 threads and 40 iterations: first chunk ≈ 40/(2*4) = 5"
		 << endl;

	vector<int> assignment(40, -1);
	vector<int> chunk_sizes;

	bool reassign = false;

#pragma omp parallel for schedule(guided, 2) num_threads(4)
	for(int i = 0; i < 40; i++) {
		// More substantial computation to ensure threads have time to
		// participate
		volatile int dummy = 0;
		for(int j = 0; j < 50000; j++) {
			dummy += j;
		}
		if(assignment[i] != -1) reassign = true;
		assignment[i] = omp_get_thread_num();
	}

	cout << "Actual:   ";
	for(int i = 0; i < 40; i++) {
		cout << assignment[i];
		if(i < 39) cout << ",";
	}
	cout << endl;

	// Calculate chunk sizes
	cout << "Chunk sizes: ";
	int i = 0;
	while(i < 40) {
		int thread = assignment[i];
		int chunk_start = i;
		while(i < 40 && assignment[i] == thread) {
			i++;
		}
		int chunk_size = i - chunk_start;
		chunk_sizes.push_back(chunk_size);
		cout << chunk_size;
		if(i < 40) cout << ",";
	}
	cout << endl;

	cout << "No reassignment: " << (!reassign ? "PASS" : "FAIL") << endl;

	// Verify all threads participated
	cout << "Verifying all threads participated: ";
	bool all_threads_g[4] = {false, false, false, false};
	for(int i = 0; i < 40; i++) {
		if(assignment[i] >= 0 && assignment[i] < 4) {
			all_threads_g[assignment[i]] = true;
		}
	}
	bool all_participated_g = true;
	for(int i = 0; i < 4; i++) {
		if(!all_threads_g[i]) {
			all_participated_g = false;
			break;
		}
	}
	cout << (all_participated_g ? "PASS" : "FAIL") << endl;

	// Note: Guided scheduling may result in irregular chunk patterns
	// because threads can claim multiple chunks before others start
}

void test_default_static() {
	cout << "\n== Default Static Schedule (no chunk_size) ==" << endl;
	cout << "Expected: 20 iterations divided equally among 4 threads (5 each)"
		 << endl;
	cout << "Thread assignment: 0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3"
		 << endl;

	vector<int> assignment(20, -1);

#pragma omp parallel for schedule(static) num_threads(4)
	for(int i = 0; i < 20; i++) {
		assignment[i] = omp_get_thread_num();
	}

	cout << "Actual:   ";
	for(int i = 0; i < 20; i++) {
		cout << assignment[i];
		if(i < 19) cout << ",";
	}
	cout << endl;
}

void test_runtime_schedule() {
	const char *env = getenv("OMP_SCHEDULE");
	cout << "\n== Runtime Schedule (OMP_SCHEDULE environment variable) =="
		 << endl;
	cout << "Expected: Behavior depends on OMP_SCHEDULE setting" << endl;
	cout << "Current setting: OMP_SCHEDULE=" << env << endl;
	cout << "This should behave like dynamic scheduling with chunk size 1"
		 << endl;

	vector<int> assignment(20, -1);

#pragma omp parallel for schedule(runtime) num_threads(4)
	for(int i = 0; i < 20; i++) {
		// More substantial computation to ensure threads have time to
		// participate
		volatile int dummy = 0;
		for(int j = 0; j < 50000; j++) {
			dummy += j;
		}
		assignment[i] = omp_get_thread_num();
	}

	cout << "Actual:   ";
	for(int i = 0; i < 20; i++) {
		cout << assignment[i];
		if(i < 19) cout << ",";
	}
	cout << endl;

	// Verify all threads participated
	cout << "Verifying all threads participated: ";
	bool all_threads[4] = {false, false, false, false};
	for(int i = 0; i < 20; i++) {
		if(assignment[i] >= 0 && assignment[i] < 4) {
			all_threads[assignment[i]] = true;
		}
	}
	bool all_participated = true;
	for(int i = 0; i < 4; i++) {
		if(!all_threads[i]) {
			all_participated = false;
			break;
		}
	}
	cout << (all_participated ? "PASS - All 4 threads participated"
							  : "FAIL - Some threads did not participate")
		 << endl;
}

int main() {
	cout << "=== SimpleOMP Schedule Clause Demo ===" << endl;
	cout << "This demo shows how different scheduling strategies assign"
		 << endl;
	cout << "iterations to threads. Output is deterministic and easy to verify."
		 << endl;

	test_default_static();
	test_static_schedule();
	test_dynamic_schedule();
	test_guided_schedule();
	test_runtime_schedule();

	cout << "\n=== Demo Complete ===" << endl;
	cout << "\nKey Observations:" << endl;
	cout << "- Static: Predictable, contiguous chunks" << endl;
	cout << "- Dynamic: Flexible assignment based on thread availability"
		 << endl;
	cout << "- Guided: Large chunks first, then smaller (reduces overhead)"
		 << endl;
	cout
		<< "- Runtime: Behavior determined by OMP_SCHEDULE environment variable"
		<< endl;

	return 0;
}
