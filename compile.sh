#!/bin/bash
nvcc -Xcompiler -fPIC -shared -o zoutPlusQsgd.so zeroOutPlusQsgdPy.cu
nvcc -Xcompiler -fPIC -shared -o zoutPlusQsgdDecom.so zeroOutPlusQsgdDecomPy.cu