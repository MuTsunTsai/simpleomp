#include <chrono>
#include <iostream>

using namespace std;
using namespace chrono;

int main() {
	cout << "=== OpenMP If Clause Demo ===" << endl << endl;

	int n = 100;

	// Test 1: n=100, should execute serially
	auto start = high_resolution_clock::now();
#pragma omp parallel for if(n > 1000) num_threads(8)
	for(int i = 0; i < 100000000; i++) {
		volatile int x = i * i;
	}
	auto end = high_resolution_clock::now();
	auto time = duration_cast<milliseconds>(end - start).count();
	cout << "n=100 (serial): " << time << " ms (" << time / 100.0
		 << "ms per iteration)" << endl;

	// Test 2: n=5000, should execute in parallel
	n = 5000;
	start = high_resolution_clock::now();
#pragma omp parallel for if(n > 1000) num_threads(8)
	for(int i = 0; i < 100000000; i++) {
		volatile int x = i * i;
	}
	end = high_resolution_clock::now();
	time = duration_cast<milliseconds>(end - start).count();
	cout << "n=5000 (parallel): " << time << " ms (" << time / 5000.0
		 << "ms per iteration)" << endl;

	return 0;
}