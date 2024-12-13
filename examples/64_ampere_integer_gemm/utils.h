// Helper methods to check for errors
#include "helper.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////
#define UNDERLINE "\033[4m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define RESET "\033[0m"

#define RANDOM_BUFFER false

#define CREATE_OUTPUT_TENSOR(tensor_d) \
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn()); \
  cutlass::reference::host::TensorFill( \
      tensor_d.host_view()); \
  tensor_d.sync_device(); 

#define INITIALIZE_GEMM(Gemm, tensor_d) \
  std::cout << "---------------------------------------------" << std::endl;        \
  std::cout << "+ Running " << YELLOW << UNDERLINE << #Gemm << RESET << std::endl;  \
  typename Gemm::Arguments arguments{problem_size,                                  \
                                     tensor_a.device_ref(),                         \
                                     tensor_b.device_ref(),                         \
                                     tensor_c.device_ref(),                         \
                                     tensor_d.device_ref(),                         \
                                     {alpha, beta}};                                \
  size_t workspace_size = Gemm::get_workspace_size(arguments);                      \
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);            \
  Gemm gemm_op;

template <typename ElementOutput, typename LayoutOutput>
bool CheckTensorEqual(cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_d_native,
                      cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_d) {
  tensor_d_native.sync_host();
  tensor_d.sync_host();
  return cutlass::reference::host::TensorEquals(
      tensor_d_native.host_view(), tensor_d.host_view());
}

inline void CheckGPUStatus(cudaError_t status, std::string_view msg) {
  if (status != cudaSuccess) {
    printf("[%s %s] ", msg.data(), cudaGetErrorString(status));
  }
  return;
}

template <typename T, typename Layout>
void PrintTensor(cutlass::HostTensor<T, Layout> &tensor, size_t elements=3) {
  tensor.sync_host();
  for (int i = 0; i < elements; i++) {
    std::cout << tensor.at({0, i}) << " ";
  }
  std::cout << std::endl;
}

/// Simple function to initialize a buffer to arbitrary small integers.
template <typename T>
cudaError_t InitializeBuffer(T *buffer, int buffer_size, int seed = 0) {

  for (int i = 0; i < buffer_size; ++i) {
    int offset = i;

    // Generate arbitrary elements.
    T value;
    if constexpr(RANDOM_BUFFER) {
      int const k = 16807;
      int const m = 16;
      value = static_cast<T>(((offset + seed) * k % m) - m / 2);
    } else {
      value = T(1);
    }

    buffer[offset] = value;
  }

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;
  
  Options():
    help(false),
    problem_size({5120, 4096, 4096}),
    batch_count(1),
    reference_check(true),
    iterations(20),
    alpha(1),
    beta() { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "14_ampere_tf32_tensorop_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/64_ampere_integer_gemm/ampere_integer_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / (runtime_s + 1e-9);
  }
};

/// Result structure
struct Result {
  Options* options_ptr;

  float runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;
  cudaEvent_t events[2];

  //
  // Methods
  //

  // TODO: No EXCEPTION during RAII must be ensured!
  Result(
    Options* options_ptr,
    float runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    options_ptr(options_ptr), runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) {
      for (auto & event : events) {
        error = cudaEventCreate(&event);
        CheckGPUStatus(error, "cudaEventCreate() failed: ");
      }

      error = cudaEventRecord(events[0]);
      CheckGPUStatus(error, "cudaEventRecord() failed: ");
    }
  
  ~Result() {
    // Record an event when the GEMMs are complete
    error = cudaEventRecord(events[1]);
    CheckGPUStatus(error, "cudaEventRecord() failed: ");

    // Wait for work on the device to complete.
    error = cudaEventSynchronize(events[1]);
    CheckGPUStatus(error, "cudaEventSynchronize() failed: ");

    // Measure elapsed runtime
    float runtime_ms_float = 0;
    error = cudaEventElapsedTime(&runtime_ms_float, events[0], events[1]);
    CheckGPUStatus(error, "cudaEventElapsedTime() failed: ");

    runtime_ms = static_cast<double>(runtime_ms_float) / static_cast<double>(options_ptr->iterations);
    gflops = options_ptr->gflops(runtime_ms / 1000.0);

    for (auto & event : events) {
      CheckGPUStatus(cudaEventDestroy(event), "cudaEventDestroy() failed: ");
    }

    std::cout << "Runtime: " << runtime_ms << " ms" << std::endl;
    std::cout << " GOPs: " << gflops << std::endl;
  }
};
