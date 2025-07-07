```shell


mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_ENABLE_TESTS=OFF -DCUTLASS_ENABLE_EXAMPLES=ON -DCUTLASS_ENABLE_LIBRARY=OFF -DCUTLASS_ENABLE_PROFILER=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j 16
./examples/64_ampere_integer_gemm/64_ampere_integer_gemm
ncu -o profile128 --import-source 1 --set full ./examples/64_ampere_integer_gemm/64_ampere_integer_gemm
```