# COMPSO_Compressor

## The GPU implementation of "COMPSO: Optimizing Gradient Compression for Distributed Training with Second-Order Optimizers"

## Environment Requirements
- Linux OS with NVIDIA GPUs
- Git >= 2.15
- CMake >= 3.21
- Cuda Toolkit >= 11.0
- GCC >= 7.3.0

## Compile
```bash
$ git clone https://github.com/Presciman/COMPSO_Compressor.git
$ bash compile.sh
```
### This will generate .so files.

### Note: If you need to setup CUDA and GCC environment variables, modify "setup_env_cuda.sh"


## Usage
### compressor_decompressor.py gives an example of integrating COMPSO GPU kernel with python.

## TODO
### We are developing the compressor into a library for easier integration into python scripts. 