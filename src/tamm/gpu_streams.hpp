#pragma once

#include "tamm/errors.hpp"
#include <map>
#include <sstream>
#include <vector>

#if defined(USE_CUDA)
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
#elif defined(USE_DPCPP)
#include "sycl_device.hpp"
#include <oneapi/mkl/blas.hpp>
#endif

namespace tamm {

#if defined(USE_HIP)
using gpuStream_t     = hipStream_t;
using gpuEvent_t      = hipEvent_t;
using gpuBlasHandle_t = rocblas_handle;
using gpuMemcpyKind   = hipMemcpyKind;
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice

#define HIP_CHECK(FUNC)                                                                           \
  do {                                                                                            \
    hipError_t err_ = (FUNC);                                                                     \
    if(err_ != hipSuccess) {                                                                      \
      std::ostringstream msg;                                                                     \
      msg << "HIP Error: " << hipGetErrorString(err_) << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                           \
      throw std::runtime_error(msg.str());                                                        \
    }                                                                                             \
  } while(0)

#define ROCBLAS_CHECK(FUNC)                                                                      \
  do {                                                                                           \
    rocblas_status err_ = (FUNC);                                                                \
    if(err_ != rocblas_status_success) {                                                         \
      std::ostringstream msg;                                                                    \
      msg << "ROCBLAS Error: " << rocblas_status_to_string(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                              \
      throw std::runtime_error(msg.str());                                                       \
    }                                                                                            \
  } while(0)
#endif // USE_HIP

#if defined(USE_CUDA)
using gpuStream_t     = cudaStream_t;
using gpuEvent_t      = cudaEvent_t;
using gpuBlasHandle_t = cublasHandle_t;
using gpuMemcpyKind   = cudaMemcpyKind;
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#define CUDA_CHECK(FUNC)                                                                \
  do {                                                                                  \
    cudaError_t err_ = (FUNC);                                                          \
    if(err_ != cudaSuccess) {                                                           \
      std::ostringstream msg;                                                           \
      msg << "CUDA Error: " << cudaGetErrorString(err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                     \
      throw std::runtime_error(msg.str());                                              \
    }                                                                                   \
  } while(0)

#define CUBLAS_CHECK(FUNC)                                                                        \
  do {                                                                                            \
    cublasStatus_t err_ = (FUNC);                                                                 \
    if(err_ != CUBLAS_STATUS_SUCCESS) {                                                           \
      std::ostringstream msg;                                                                     \
      msg << "CUBLAS Error: " << /*cublasGetStatusString*/ (err_) << ", at " << __FILE__ << " : " \
          << __LINE__ << std::endl;                                                               \
      throw std::runtime_error(msg.str());                                                        \
    }                                                                                             \
  } while(0)
#endif // USE_CUDA

#if defined(USE_DPCPP)
using gpuStream_t   = sycl::queue;
using gpuEvent_t    = sycl::event;
using gpuMemcpyKind = int;
#define gpuMemcpyHostToDevice 0
#define gpuMemcpyDeviceToHost 1
#define gpuMemcpyDeviceToDevice 2

auto sycl_asynchandler = [](sycl::exception_list exceptions) {
  for(std::exception_ptr const& e: exceptions) {
    try {
      std::rethrow_exception(e);
    } catch(sycl::exception const& ex) {
      std::cout << "Caught asynchronous SYCL exception:" << std::endl
                << ex.what() << ", SYCL code: " << ex.code() << std::endl;
    }
  }
};

#define ONEMKLBLAS_CHECK(FUNC)                                                         \
  do {                                                                                 \
    try {                                                                              \
      (FUNC)                                                                           \
    } catch(oneapi::mkl::exception const& ex) {                                        \
      std::ostringstream msg;                                                          \
      msg << "oneMKL Error: " << ex.what() << ", at " << __FILE__ << " : " << __LINE__ \
          << std::endl;                                                                \
      throw std::runtime_error(msg.str());                                             \
    }                                                                                  \
  } while(0)
#endif // USE_DPCPP

static inline void getDeviceCount(int* id) {
#if defined(USE_CUDA)
  CUDA_CHECK(cudaGetDeviceCount(id));
#elif defined(USE_HIP)
  HIP_CHECK(hipGetDeviceCount(id));
#elif defined(USE_DPCPP)
  syclGetDeviceCount(id);
#endif
}

static inline void gpuMemGetInfo(size_t* free, size_t* total) {
#if defined(USE_CUDA)
  cudaMemGetInfo(free, total);
#elif defined(USE_HIP)
  hipMemGetInfo(free, total);
#elif defined(USE_DPCPP)
  syclMemGetInfo(free, total);
#endif
}

static inline void gpuSetDevice(int active_device) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaSetDevice(active_device));
#elif defined(USE_HIP)
  HIP_CHECK(hipSetDevice(active_device));
#elif defined(USE_DPCPP)
  syclSetDevice(active_device);
#endif
}

static inline void gpuGetDevice(int* active_device) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaGetDevice(active_device));
#elif defined(USE_HIP)
  HIP_CHECK(hipGetDevice(active_device));
#elif defined(USE_DPCPP)
  syclGetDevice(active_device);
#endif
}

template<typename T>
static void gpuMemcpyAsync(T* dst, const T* src, size_t count, gpuMemcpyKind kind,
                           gpuStream_t& stream) {
#if defined(USE_DPCPP)
  if(kind == gpuMemcpyDeviceToDevice) { stream.copy(src, dst, count); }
  else { stream.memcpy(dst, src, count * sizeof(T)); }
#elif defined(USE_CUDA)
  CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
#elif defined(USE_HIP)
  HIP_CHECK(hipMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
#endif
}

static inline void gpuMemsetAsync(void*& ptr, size_t sizeInBytes, gpuStream_t stream) {
#if defined(USE_DPCPP)
  stream.memset(ptr, 0, sizeInBytes);
#elif defined(USE_HIP)
  hipMemsetAsync(ptr, 0, sizeInBytes, stream);
#elif defined(USE_CUDA)
  cudaMemsetAsync(ptr, 0, sizeInBytes, stream);
#endif
}

static inline void gpuStreamSynchronize(gpuStream_t stream) {
#if defined(USE_DPCPP)
  stream.wait();
#elif defined(USE_HIP)
  hipStreamSynchronize(stream);
#elif defined(USE_CUDA)
  cudaStreamSynchronize(stream);
#endif
}

class GPUStreamPool {
protected:
  bool _initialized{false};

  int _ngpus{0};
  // Active GPU set by a given MPI-rank from execution context ctor
  int _active_device{0};

  // Map of GPU-IDs and stream
  std::map<int, gpuStream_t*> _devID2Stream;

#if defined(USE_CUDA) || defined(USE_HIP)
  // Map of GPU-IDs and blashandle
  std::map<int, gpuBlasHandle_t*> _devID2Handle;
#endif

private:
  GPUStreamPool() {
    getDeviceCount(&_ngpus);
    // EXPECTS_STR((_ngpus == 1), "Error: More than 1 GPU-device found per rank!");

    for(int devID = 0; devID < _ngpus; devID++) {
      gpuSetDevice(devID);

      // populate gpu-streams, gpu-blas handles per GPU
      gpuStream_t* _devStream = nullptr;
#if defined(USE_CUDA)
      _devStream = new cudaStream_t;
      CUDA_CHECK(cudaStreamCreateWithFlags(_devStream, cudaStreamNonBlocking));

      gpuBlasHandle_t* _devHandle = new gpuBlasHandle_t;
      CUBLAS_CHECK(cublasCreate(_devHandle));
      CUBLAS_CHECK(cublasSetStream(*_devHandle, *_devStream));
      _devID2Handle[devID] = _devHandle;
#elif defined(USE_HIP)
      _devStream = new hipStream_t;
      HIP_CHECK(hipStreamCreateWithFlags(_devStream, hipStreamNonBlocking));

      gpuBlasHandle_t* _devHandle = new gpuBlasHandle_t;
      ROCBLAS_CHECK(rocblas_create_handle(_devHandle));
      ROCBLAS_CHECK(rocblas_set_stream(*_devHandle, *_devStream));
      _devID2Handle[devID] = _devHandle;
#elif defined(USE_DPCPP)
      _devStream = new sycl::queue(*sycl_get_context(devID), *sycl_get_device(devID),
                                   sycl_asynchandler,
                                   sycl::property_list{sycl::property::queue::in_order{}});
#endif

      _devID2Stream[devID] = _devStream;
    }

    _initialized = (_ngpus == 1) ? true : false;
  }

  ~GPUStreamPool() {
    _initialized = false;

    for(int devID = 0; devID < _ngpus; devID++) {
      gpuSetDevice(devID);
      gpuStream_t* _devStream = _devID2Stream[devID];
#if defined(USE_CUDA)
      CUDA_CHECK(cudaStreamDestroy(*_devStream));
      CUBLAS_CHECK(cublasDestroy(*_devID2Handle[devID]));
#elif defined(USE_HIP)
      HIP_CHECK(hipStreamDestroy(*_devStream));
      ROCBLAS_CHECK(rocblas_destroy_handle(*_devID2Handle[devID]));
#elif defined(USE_DPCPP)
      delete _devStream;
#endif
      _devStream = nullptr;
    }
  }

public:
  /// sets an active device for getting streams and blas handles
  void set_device(int device) {
    if(!_initialized) {
      _active_device = device;
      gpuSetDevice(_active_device);
      _initialized = true;
    }
  }

  /// Returns a GPU stream
  gpuStream_t& getStream() { return *(_devID2Stream[_active_device]); }

#if !defined(USE_DPCPP)
  /// Returns a GPU BLAS handle that is valid only for the CUDA and HIP builds
  gpuBlasHandle_t& getBlasHandle() { return *(_devID2Handle[_active_device]); }
#endif

  /// Returns the instance of device manager singleton.
  inline static GPUStreamPool& getInstance() {
    static GPUStreamPool d_m{};
    return d_m;
  }

  GPUStreamPool(const GPUStreamPool&)            = delete;
  GPUStreamPool& operator=(const GPUStreamPool&) = delete;
  GPUStreamPool(GPUStreamPool&&)                 = delete;
  GPUStreamPool& operator=(GPUStreamPool&&)      = delete;
};

// This API needs to be defined after the class GPUStreamPool since the classs
// is only declared and defined before this method
} // namespace tamm
