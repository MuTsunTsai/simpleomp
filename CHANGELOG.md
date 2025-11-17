# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.1.0]: https://github.com/MuTsunTsai/simpleomp/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/MuTsunTsai/simpleomp/releases/tag/v1.0.0
