#!/bin/bash -l

export CUDA_HOME=/lus/grand/projects/sbi-fair/spack_polaris/opt/spack/linux-sles15-zen2/gcc-9.3.0/cuda-11.0.3-s6mxqre76aiqvecmepnvoiaqkws3xs7j
export PATH=/lus/grand/projects/sbi-fair/spack_polaris/opt/spack/linux-sles15-zen2/gcc-9.3.0/cuda-11.0.3-s6mxqre76aiqvecmepnvoiaqkws3xs7j/bin:$PATH
export PATH=/grand/sbi-fair/baixi/conda_envs/bert-polaris/bin:$PATH
export PATH=/lus/grand/projects/sbi-fair/spack_polaris/opt/spack/linux-sles15-zen/gcc-7.5.0/gcc-9.3.0-iccip3nccyoeactjlahixbvmml5wg4pg/bin:$PATH
export LD_LIBRARY_PATH=/lus/grand/projects/sbi-fair/spack_polaris/opt/spack/linux-sles15-zen/gcc-7.5.0/gcc-9.3.0-iccip3nccyoeactjlahixbvmml5wg4pg/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/lus/grand/projects/sbi-fair/spack_polaris/opt/spack/linux-sles15-zen2/gcc-9.3.0/cuda-11.0.3-s6mxqre76aiqvecmepnvoiaqkws3xs7j/lib64:$LD_LIBRARY_PATH

