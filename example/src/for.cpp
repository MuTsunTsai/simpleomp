#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

int main() {
	cout << "=== OpenMP Parallel For Demo ===" << endl
			  << endl;

	// Large array for computation
	vector<double> data(10000);
	vector<double> buffer(10000);

	// Initialize array with some values
	for(int i = 0; i < 10000; i++) {
		data[i] = static_cast<double>(i) * 0.5;
	}

	int n_threads = 16;

	// Start timing
	auto start = high_resolution_clock::now();

// Compute-intensive operation: calculate complex mathematical expressions
// Each iteration performs multiple floating-point operations
#pragma omp parallel for num_threads(n_threads)
	for(int i = 0; i < 10000; i++) {
		double temp = data[i];
		// Perform heavy computation: polynomial evaluation
		for(int j = 1; j < 200000; j++) {
			temp = temp * 1.0001 + 0.0001;
			temp = sqrt(temp + j);
		}
		buffer[i] = temp;
	}

	// End timing
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(end - start);

	cout << "Computation completed in " << duration.count() << " ms" << endl;

	// Sum
	double result = 0.0;
	for(int i = 0; i < 10000; i++) {
		result += data[i];
	}

	cout << "Result is " << result << endl;

	return 0;
}
