# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2025-01-20

### Added
- Support for OpenMP cancellation constructs:
  - **`#pragma omp cancel`** - Request cancellation of parallel regions or loops
    - Supports cancellation types: `parallel`, `for`, `sections`, `taskgroup`
    - Cancellation propagates to all threads in the same team
    - Returns boolean indicating whether the thread should terminate
  - **`#pragma omp cancellation point`** - Check for cancellation requests
    - Allows threads to respond to cancellation requests at specific points
    - Essential for implementing early termination in parallel algorithms
  - **Environment control**: `OMP_CANCELLATION` environment variable (default: enabled)
  - Example: [example/src/cancel.cpp](example/src/cancel.cpp)
- OpenMP Runtime Library API header file:
  - **New header**: [include/omp.h](include/omp.h) - Standard OpenMP API declarations (OpenMP 5.0 compliant)
  - **New implementation**: [src/omp_runtime.cpp](src/omp_runtime.cpp)
  - **Thread management**: `omp_set_num_threads()`, `omp_get_num_threads()`, `omp_get_thread_num()`, `omp_get_num_procs()`, etc.
  - **Lock mechanisms**: Full implementation of `omp_lock_t` and `omp_nest_lock_t` using `ncnn::Mutex`
  - **Timing functions**: `omp_get_wtime()`, `omp_get_wtick()` using Emscripten high-resolution timers
  - **Schedule control**: `omp_set_schedule()`, `omp_get_schedule()`
  - **Hierarchy queries**: `omp_get_level()`, `omp_get_ancestor_thread_num()`, `omp_get_team_size()`, etc.
  - **Stub implementations**: Specification-compliant stubs for unsupported features (affinity, etc.)
  - Example: [example/src/locks.cpp](example/src/locks.cpp)
- Data-sharing clauses demonstration:
  - **New example**: [example/src/data_sharing.cpp](example/src/data_sharing.cpp)
  - Demonstrates `private`, `shared`, `firstprivate`, `lastprivate` clauses
  - Shows how Clang handles variable scoping at compile-time
  - Educational example explaining compiler-level vs runtime-level features

### Technical Details
- **New file**: [src/kmp_cancel.cpp](src/kmp_cancel.cpp) (~150 lines)
  - Implements `__kmpc_cancel()`, `__kmpc_cancellation_point()`, `__kmpc_cancel_barrier()`
  - Uses global map to track cancellation state per team
  - Thread-safe cancellation flag management with atomic operations
  - Supports all OpenMP cancellation construct types via bitmask flags
  - Environment variable control: reads `OMP_CANCELLATION` at runtime (defaults to "true")
- **New file**: [include/omp.h](include/omp.h) (~200 lines)
  - Complete OpenMP API declarations following OpenMP 5.0 specification
  - Type definitions: `omp_lock_t`, `omp_nest_lock_t`, `omp_sched_t`, etc.
  - Function declarations for all standard OpenMP runtime functions
  - Replaces need for manual `extern "C"` declarations in user code
- **New file**: [src/omp_runtime.cpp](src/omp_runtime.cpp) (~400 lines)
  - Implements all OpenMP runtime API functions
  - Lock types use internal `ncnn::Mutex` with proper nesting support
  - Timing functions use `emscripten_get_now()` for microsecond precision
  - Hierarchy functions support nested parallel regions (currently 1 level)
  - Provides informative stubs for features not yet implemented
- **Example enhancement**: [example/src/data_sharing.cpp](example/src/data_sharing.cpp)
  - Comprehensive demonstration of all data-sharing clause behaviors
  - Shows memory addresses to illustrate private vs shared variables
  - Documents which clauses are compiler-handled vs runtime-handled
  - Clarifies common misconceptions about OpenMP variable scoping

## [1.4.0] - 2025-01-19

### Added
- Support for OpenMP atomic operations (`#pragma omp atomic`):
  - **Arithmetic operations**: `add`, `sub`, `mul`, `div` (integers and floating-point)
  - **Bitwise operations**: `and`, `or`, `xor` (integer types only)
  - **Comparison operations**: `min`, `max` (integers and floating-point)
  - **Memory operations**: `read`, `write` (atomic load/store)
  - **Type support**: 8/16/32/64-bit integers (signed/unsigned), `float`, `double`
  - Example: [example/src/atomic.cpp](example/src/atomic.cpp)
- Support for `nowait` clause (compiler-handled, demonstration example only):
  - Allows skipping implicit barriers at the end of worksharing constructs
  - Works with `#pragma omp for`, `#pragma omp single`, etc.
  - Example: [example/src/nowait.cpp](example/src/nowait.cpp)

### Technical Details
- **New file**: [src/kmp_atomic.cpp](src/kmp_atomic.cpp) (~420 lines)
  - Implements LLVM libomp atomic ABI functions for various operations and types
  - **Integer operations**: Use native atomic `fetch_*` hardware instructions
  - **Floating-point operations**: Use compare-and-swap (CAS) loops for non-native operations
  - **Read/Write operations**: Use atomic `load`/`store` with sequential consistency
  - Covers 32 functions: 8 operations × 4 types (int32/uint32/int64/uint64/float/double)
  - Note: `atomic capture` is not yet implemented
- **`nowait` clause**: Compiler-level feature requiring no runtime implementation
  - Clang omits `__kmpc_barrier()` calls when `nowait` is present in LLVM IR
  - SimpleOMP inherently supports this through existing barrier infrastructure
  - Demonstration example shows performance benefits of skipping implicit barriers

## [1.3.0] - 2025-01-18

### Added
- Support for OpenMP `schedule` clause with dynamic and guided scheduling strategies:
  - **`schedule(dynamic [, chunk_size])`** - Dynamic work distribution
    - Lock-free atomic chunk allocation using compare-and-swap operations
    - Ideal for workloads with irregular iteration costs
    - Configurable chunk size (defaults to 1)
  - **`schedule(guided [, min_chunk])`** - Guided self-scheduling
    - Exponentially decreasing chunk sizes for better load balancing
    - Starts with `remaining_iterations / (2 * num_threads)`, decreases to `min_chunk`
    - Balances overhead reduction with load distribution
  - **`schedule(runtime)`** - Runtime schedule selection
    - Reads scheduling strategy from `OMP_SCHEDULE` environment variable
    - Supports formats: `"static,N"`, `"dynamic,N"`, `"guided,N"`
    - Allows dynamic scheduling decisions without recompilation
  - **`schedule(static, chunk_size)`** - Enhanced static chunked scheduling
    - Round-robin chunk distribution across threads
    - Proper support for arbitrary loop strides (not just `i++`)
- New comprehensive example: [example/src/schedule.cpp](example/src/schedule.cpp)
  - Demonstrates all scheduling strategies with visual iteration distribution
  - Includes performance comparison between static, dynamic, and guided schedules

### Technical Details
- **New file**: [src/kmp_dispatch.cpp](src/kmp_dispatch.cpp) (~900 lines)
  - Implements LLVM libomp dispatch ABI: `__kmpc_dispatch_init_*`, `__kmpc_dispatch_next_*`, `__kmpc_dispatch_deinit`
  - Supports 4 integer types: `int32_t`, `uint32_t`, `int64_t`, `uint64_t`
  - Uses deferred deletion pattern to prevent use-after-free in multi-threaded environments
  - Generation counter mechanism to detect loop reinitialization and prevent race conditions
  - Thread-local storage for tracking expected generation per thread
  - Lock-free algorithms for dynamic chunk allocation with atomic operations
  - Guided scheduling uses exponential decay formula: `chunk = max((upper - current) / (2 * num_threads), min_chunk)`
- **Enhanced**: `__kmpc_for_static_init_4` in [src/simpleomp.cpp](src/simpleomp.cpp)
  - Added support for `schedule(static, chunk)` with proper round-robin distribution
  - Correctly handles arbitrary loop increments/decrements (stride parameter)
  - Extracts base schedule type to ignore modifier flags

## [1.2.0] - 2025-01-17

### Added
- Support for OpenMP synchronization constructs:
  - **`#pragma omp barrier`** - Thread barrier synchronization
    - Implemented `__kmpc_barrier()` function using generation counter pattern
    - Ensures all threads reach the synchronization point before continuing
    - Example: [example/src/barrier.cpp](example/src/barrier.cpp)
  - **`#pragma omp critical`** - Critical sections for mutual exclusion
    - Implemented `__kmpc_critical()` and `__kmpc_end_critical()` functions
    - Supports named critical sections using mutex map
    - Example: [example/src/critical.cpp](example/src/critical.cpp)
  - **`#pragma omp master`** - Master thread-only execution regions
    - Implemented `__kmpc_master()` and `__kmpc_end_master()` functions
    - Simple thread ID check for master thread (thread 0)
    - Example: [example/src/master.cpp](example/src/master.cpp)
  - **`#pragma omp single`** - Single thread execution regions
    - Implemented `__kmpc_single()` and `__kmpc_end_single()` functions
    - Uses map to track first arriving thread per location
    - Example: [example/src/single.cpp](example/src/single.cpp)

### Technical Details
- **Barrier implementation**: Uses mutex, condition variable, and generation counter to handle multiple barrier calls correctly
- **Critical sections**: Each named critical section has its own mutex stored in a thread-safe map
- **Master construct**: Lightweight implementation with simple thread number comparison
- **Single construct**: Tracks execution state per source location to ensure only one thread executes the region
- All synchronization primitives properly integrate with the existing thread pool architecture


## [1.1.0] - 2025-01-17

### Added
- Support for OpenMP `if` clause to conditionally enable/disable parallelization
  - Implemented `__kmpc_serialized_parallel()` function for serial execution path
  - Implemented `__kmpc_end_serialized_parallel()` function for cleanup
  - Example usage: `#pragma omp parallel for if(n > 1000) num_threads(8)`
- New example demonstrating `if` clause usage ([example/src/if.cpp](example/src/if.cpp))
- Example index page ([example/index.html](example/index.html)) for easy navigation between examples

### Changed
- Exported `tls_num_threads` and `tls_thread_num` from `simpleomp.cpp` for use in other compilation units
- Updated coding standards in CLAUDE.md: all code and comments must be written in English

### Technical Details
- The `if` clause allows runtime decisions about parallelization based on conditions
- When the condition evaluates to false, the code executes serially without thread creation overhead
- When the condition evaluates to true, normal parallel execution occurs

## [1.0.0] - 2025-01-14

### Added
- Initial release of SimpleOMP
- Basic OpenMP runtime for Emscripten/WebAssembly
- Support for `#pragma omp parallel for num_threads(N)`
- Static loop scheduling with `__kmpc_for_static_init_*` functions
- Thread pool management using Web Workers
- Example demonstrating parallel for loops
- Comprehensive build system with Makefile
- CI/CD pipeline for automated releases

### Technical Details
- Based on LLVM libomp ABI for Clang compatibility
- Thread-local storage for thread management
- Ring buffer task queue for efficient work distribution
- Derived from Tencent NCNN threading implementation

[1.5.0]: https://github.com/MuTsunTsai/simpleomp/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/MuTsunTsai/simpleomp/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/MuTsunTsai/simpleomp/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/MuTsunTsai/simpleomp/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/MuTsunTsai/simpleomp/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MuTsunTsai/simpleomp/releases/tag/v1.0.0
