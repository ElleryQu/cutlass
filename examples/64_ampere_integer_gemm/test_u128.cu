#include <iostream>

#define COMP_FATAL(MSG) \
    static_assert(0, MSG)

#ifndef PRINT
#define PRINT(MSG)                                                  \
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0    \
        && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)   \
    {                                                               \
        printf MSG ;                                                \
    }
#endif

template <typename T>
struct CallTracker {
    static bool called;
};

template <typename T>
bool CallTracker<T>::called = false;

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/kernel/default_gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/thread/mma.h"
#include "cutlass/functional.h"

#include "cutlass/util/host_tensor.h"

using IntT = __uint128_t;
// using IntT = uint64_t;

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = IntT;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = IntT;                        // <- data type of elements in input matrix A
using ElementInputB = IntT;                        // <- data type of elements in input matrix B
using ElementOutput = IntT;                        // <- data type of elements in output matrix D
using ElementInputC = ElementOutput;          // <- data type of elements in input matrix C

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix B, Row Major for Matrix A and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using OperatorClass = cutlass::arch::OpClassSimt;
using ArchTag = cutlass::arch::Sm50;  // Ampere architecture

using OutputOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,  // Data-type of output matrix D
    16 / sizeof(ElementOutput),  // Data-type of epilogue operation
    ElementAccumulator,  // Data-type of accumulator
    ElementOutput,  // Data-type of output matrix D
    cutlass::epilogue::thread::ScaleType::Nothing>;  // Scale type for epilogue operation

// Device.
using CutlassDefaultGemmSimt = cutlass::gemm::device::Gemm<ElementInputA,          // Data-type of A matrix
                                                LayoutInputA,               // Layout of A matrix
                                                ElementInputB,              // Data-type of B matrix
                                                LayoutInputB,               // Layout of B matrix
                                                ElementAccumulator,         // Data-type of C matrix
                                                LayoutOutput,
                                                ElementOutput,
                                                OperatorClass,
                                                ArchTag>;       // Layout of C matrix
using CutlassGemmSimt = cutlass::gemm::device::Gemm<ElementInputA,          // Data-type of A matrix
                                                LayoutInputA,               // Layout of A matrix
                                                ElementInputB,              // Data-type of B matrix
                                                LayoutInputB,               // Layout of B matrix
                                                ElementAccumulator,         // Data-type of C matrix
                                                LayoutOutput,
                                                ElementOutput,
                                                OperatorClass,
                                                ArchTag,
                                                CutlassDefaultGemmSimt::ThreadblockShape,
                                                CutlassDefaultGemmSimt::WarpShape,
                                                CutlassDefaultGemmSimt::InstructionShape,
                                                OutputOp>;
// using CutlassGemmSimt = CutlassDefaultGemmSimt;  // Device GEMM operation for CUTLASS

// Kernel.
using ThreadblockSwizzle = typename CutlassGemmSimt::ThreadblockSwizzle;  // Threadblock swizzling function
// using ThreadblockShape = typename cutlass::gemm::device::DefaultGemmConfiguration<
//             OperatorClass, ArchTag, ElementInputA, ElementInputB, ElementInputC,
//             ElementAccumulator>::ThreadblockShape;
using ThreadblockShape = typename CutlassGemmSimt::ThreadblockShape;  // Threadblock shape
using EpilogueOutputOp = typename CutlassGemmSimt::EpilogueOutputOp;  // Epilogue output operation
using CutlassGemmKernel = typename CutlassGemmSimt::GemmKernel;  // Kernel type for CUTLASS GEMM operation

// Mma.
using IntMma = cutlass::gemm::thread::Mma<
    cutlass::gemm::GemmShape<32, 32, 32>,
    ElementInputA, cutlass::layout::RowMajor,
    ElementInputB, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::layout::RowMajor,
    cutlass::arch::OpMultiplyAdd>;

int main()
{
    // Initialize the CUTLASS GEMM operation. Compile hang at the last step.
    {
        CutlassGemmSimt cutlass_gemm_op{};
        cutlass::gemm::GemmCoord problem_size(32, 32, 32);
        cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());  // <- Create matrix A with dimensions M x K
        cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());  // <- Create matrix B with dimensions K x N
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());  // <- Create matrix C with dimensions M x N
        cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
        tensor_a.sync_device();
        tensor_b.sync_device();
        tensor_c.sync_device();
        tensor_d.sync_device();
        typename CutlassGemmSimt::Arguments arguments{problem_size,                                  
                                     tensor_a.device_ref(),                         
                                     tensor_b.device_ref(),                         
                                     tensor_c.device_ref(),                         
                                     tensor_d.device_ref(),                         
                                     {1, 0}};                                
        size_t workspace_size = CutlassGemmSimt::get_workspace_size(arguments);                      
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        cutlass::Status status = cutlass_gemm_op.initialize(arguments, workspace.get());
        status = cutlass_gemm_op();
    }

    // // Initialize the kernel::DefaultGemm.
    // {
    //     cutlass::gemm::GemmCoord problem_size(32, 32, 32);
    //     cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(problem_size.mk());  // <- Create matrix A with dimensions M x K
    //     cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(problem_size.kn());  // <- Create matrix B with dimensions K x N
    //     cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());  // <- Create matrix C with dimensions M x N
    //     cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
    //     tensor_a.sync_device();
    //     tensor_b.sync_device();
    //     tensor_c.sync_device();
    //     tensor_d.sync_device();
    //     typename CutlassGemmSimt::Arguments arguments{problem_size,                                  
    //                                  tensor_a.device_ref(),                         
    //                                  tensor_b.device_ref(),                         
    //                                  tensor_c.device_ref(),                         
    //                                  tensor_d.device_ref(),                         
    //                                  {1, 0}};                                
    //     size_t workspace_size = CutlassGemmSimt::get_workspace_size(arguments);                      
    //     cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    //     ThreadblockSwizzle threadblock_swizzle;
    //     cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
    //         problem_size, 
    //         {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
    //         0);
    //     void *workspace_ptr = workspace.get();
    //     typename CutlassGemmKernel::Params params{
    //         problem_size,
    //         grid_shape,
    //         tensor_a.device_ref().non_const_ref(),
    //         tensor_b.device_ref().non_const_ref(),
    //         tensor_c.device_ref().non_const_ref(),
    //         tensor_d.device_ref(),
    //         typename EpilogueOutputOp::Params(),
    //         static_cast<int *>(workspace_ptr),
    //         nullptr,
    //         nullptr,
    //         nullptr
    //     };
    //     dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
    //     dim3 block(CutlassGemmKernel::kThreadCount, 1, 1);
    //     int smem_size = int(sizeof(typename CutlassGemmKernel::SharedStorage));
    //     cutlass::arch::synclog_setup();
    //     cutlass::Kernel<CutlassGemmKernel><<<grid, block, smem_size>>>(params);
    // }

    // Initialize the MMA operation. Compile pass.
    {
        IntMma mma_op{};
        typename IntMma::FragmentA x{};
        typename IntMma::FragmentB y{};
        typename IntMma::FragmentC z{};
        mma_op(z, x, y, z);
    }
    
    // cutlass::multiply_add<IntT, IntT, IntT> op;
    
    std::cout << "create IntMma object." << std::endl;
    
    return 0;
}