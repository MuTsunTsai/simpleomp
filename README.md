
# SimpleOMP

A lightweight OpenMP implementation for Emscripten, enabling basic parallel programming capabilities in WebAssembly applications.

## Overview

SimpleOMP provides a minimal OpenMP runtime for Emscripten-compiled projects. This implementation is based on the solution discussed in [emscripten-core/emscripten#13892](https://github.com/emscripten-core/emscripten/issues/13892#issuecomment-2599113825).

### Supported Features

Currently supports the following OpenMP directive:
- `#pragma omp parallel for num_threads(N)`

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

# Output will be generated at: build/dist/libsimpleomp.a
```

## Running the Example

The [example](example/) directory contains a sample project demonstrating SimpleOMP usage.

```bash
# Navigate to the example directory
cd example

# Build the example
make

# Start a local server to test (requires PNPM)
make serve
```

**Tip:** To observe the performance difference, you can comment out the `#pragma` line in [example/src/main.cpp](example/src/main.cpp), rebuild, and compare the execution time.


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