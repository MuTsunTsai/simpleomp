
# SimpleOMP

A lightweight OpenMP implementation for Emscripten, enabling basic parallel programming capabilities in WebAssembly applications.

## Overview

SimpleOMP provides a minimal OpenMP runtime for Emscripten-compiled projects. This implementation is based on the solution discussed in [emscripten-core/emscripten#13892](https://github.com/emscripten-core/emscripten/issues/13892#issuecomment-2599113825).

### Supported Features

#### Parallel Execution

| Directive/Clause | Status | Description |
|------------------|--------|-------------|
| `#pragma omp parallel for` | ✅ | Parallel for loop |
| `num_threads(N)` | ✅ | Specify thread count |
| `if(condition)` | ✅ | Conditional parallelization |
| `schedule(static[, chunk])` | ✅ | Static loop scheduling (block or round-robin) |
| `schedule(dynamic[, chunk])` | ✅ | Dynamic work distribution |
| `schedule(guided[, chunk])` | ✅ | Guided scheduling with decreasing chunk sizes |
| `schedule(runtime)` | ✅ | Runtime-determined scheduling (via `OMP_SCHEDULE`) |

#### Synchronization

| Directive/Clause | Status | Description |
|------------------|--------|-------------|
| `#pragma omp barrier` | ✅ | Thread barrier synchronization |
| `#pragma omp critical [(name)]` | ✅ | Critical section (mutual exclusion) |
| `#pragma omp master` | ✅ | Master thread-only execution |
| `#pragma omp single` | ✅ | Single thread execution |
| `#pragma omp atomic` | ❌ | Atomic operations |

#### Work Sharing

| Directive/Clause | Status | Description |
|------------------|--------|-------------|
| `#pragma omp sections` | ❌ | Separate code sections |
| `#pragma omp task` | ❌ | Task-based parallelism |

#### Data Environment

| Directive/Clause | Status | Description |
|------------------|--------|-------------|
| `reduction(op:var)` | ❌ | Reduction operations |
| `private(var)` | ❌ | Thread-private variables |
| `shared(var)` | ❌ | Shared variables |
| `firstprivate(var)` | ❌ | Initialize private from shared |
| `lastprivate(var)` | ❌ | Update shared from last iteration |


## Usage

### Prerequisites

- Emscripten toolchain
- Make

### Installation

1. Download `libsimpleomp.a` from the [releases](../../releases) page
2. Link the library when building your project
3. Add the following compilation flags: `-fopenmp -pthread`

### Example

```bash
# Compile with SimpleOMP
emcc your_code.c -fopenmp -pthread libsimpleomp.a -o output.js
```

For a complete working example, see the [example](example/) directory.

## Building from Source

To build the library from source:

```bash
# Build the library
make

# Output will be generated at: dist/libsimpleomp.a
```

## Running the Examples

The [example](example/) directory contains sample projects demonstrating SimpleOMP usage:
- **for.cpp** - Basic parallel for loop with performance comparison
- **if.cpp** - Conditional parallelization using the `if` clause
- **schedule.cpp** - Loop scheduling strategies (static/dynamic/guided) with validation
- **master.cpp** - Master thread construct demonstration
- **critical.cpp** - Critical section for mutual exclusion
- **barrier.cpp** - Barrier synchronization example
- **single.cpp** - Single thread execution construct

```bash
# Navigate to the example directory
cd example

# Build all examples
make

# Start a local server to test (requires PNPM)
make serve

# Open the URL that appears to see all examples
```

## License

This project is licensed under the MIT License - Copyright (c) 2025 Mu-Tsun Tsai.

### Third-Party Licenses

The following source files are derived from [Tencent NCNN](https://github.com/Tencent/ncnn) and are licensed under the BSD 3-Clause License:

- [src/cpu.h](src/cpu.h)
- [src/platform.h](src/platform.h)
- [src/simpleomp.cpp](src/simpleomp.cpp)

See the respective files for their copyright notices and license terms.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

This project is based on the implementation discussed in the Emscripten issue tracker. Special thanks to:
- The contributors of [emscripten-core/emscripten#13892](https://github.com/emscripten-core/emscripten/issues/13892)
- The [Tencent NCNN](https://github.com/Tencent/ncnn) project for the threading implementation code