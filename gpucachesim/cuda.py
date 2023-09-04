import click
import ctypes

# from typing import Optional


# Some constants taken from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36


def get_cuda_core_count_for_sm_version(major, minor):
    # Returns the number of CUDA cores per multiprocessor for a given
    # Compute Capability version. There is no way to retrieve that via
    # the API, so it needs to be hard-coded.
    # See _ConvertSMVer2Cores in helper_cuda.h in NVIDIA's CUDA Samples.
    return {
        (1, 0): 8,  # Tesla
        (1, 1): 8,
        (1, 2): 8,
        (1, 3): 8,
        (2, 0): 32,  # Fermi
        (2, 1): 48,
        (3, 0): 192,  # Kepler
        (3, 2): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,  # Maxwell
        (5, 2): 128,
        (5, 3): 128,
        (6, 0): 64,  # Pascal
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,  # Volta
        (7, 2): 64,
        (7, 5): 64,  # Turing
        (8, 0): 64,  # Ampere
        (8, 6): 128,
        (8, 7): 128,
        (8, 9): 128,  # Ada
        (9, 0): 128,  # Hopper
    }.get((major, minor), 0)


@click.command()
def main():
    # @click.option("--path", help="Path to materialized benchmark config")
    # @click.option("--config", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
    # @click.option("--bench", "bench_name", help="Benchmark name")
    # @click.option("--input", "input_idx", type=int, help="Input index")
    query()


class CUDAError(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def cuda_check_error(cuda, result):
    if result != CUDA_SUCCESS:
        error_str = ctypes.c_char_p()
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        error_str = "" if error_str.value is None else error_str.value.decode()
        raise CUDAError(f"CUDA error {result}: {error_str}")


def query():
    libnames = ("libcuda.so", "libcuda.dylib", "nvcuda.dll", "cuda.dll")
    for libname in libnames:
        try:
            cuda = ctypes.CDLL(libname)
        except OSError:
            continue
        else:
            break
    else:
        raise OSError("could not load any of: " + " ".join(libnames))

    nGpus = ctypes.c_int()
    name = b" " * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    freeMem = ctypes.c_size_t()
    totalMem = ctypes.c_size_t()

    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    result = cuda.cuInit(0)
    cuda_check_error(cuda, result)
    # if result != CUDA_SUCCESS:
    # cuda.cuGetErrorString(result, ctypes.byref(error_str))
    # error
    # raise RuntimeError("cuInit failed with error code %d: %s" % (result, error_str.value.decode()))
    result = cuda.cuDeviceGetCount(ctypes.byref(nGpus))
    cuda_check_error(cuda, result)
    # if result != CUDA_SUCCESS:
    #     cuda.cuGetErrorString(result, ctypes.byref(error_str))
    #     raise RuntimeError("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
    print("Found %d device(s)." % nGpus.value)
    for i in range(nGpus.value):
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        cuda_check_error(cuda, result)
        # if result != CUDA_SUCCESS:
        #     cuda.cuGetErrorString(result, ctypes.byref(error_str))
        #     error_str = "" if error_str.value is None else error_str.value.decode()
        #     raise RuntimeError(f"cuDeviceGet failed with error code {result}: {error_str}")
        print("Device: %d" % i)
        cuda_device_name = cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device)
        if cuda_device_name == CUDA_SUCCESS:
            print("  Name: %s" % (name.split(b"\0", 1)[0].decode()))

        device_compute_capability = cuda.cuDeviceComputeCapability(
            ctypes.byref(cc_major), ctypes.byref(cc_minor), device
        )
        if device_compute_capability == CUDA_SUCCESS:
            print("  Compute Capability: %d.%d" % (cc_major.value, cc_minor.value))

        result = cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device)
        if result == CUDA_SUCCESS:
            print(f"  Multiprocessors: {cores.value}")
            num_cuda_cores = cores.value * get_cuda_core_count_for_sm_version(cc_major.value, cc_minor.value)
            print(f"  CUDA Cores: {num_cuda_cores or 'unknown'}")
            result = cuda.cuDeviceGetAttribute(
                ctypes.byref(threads_per_core),
                CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
                device,
            )
            if result == CUDA_SUCCESS:
                print(f"  Concurrent threads: {cores.value * threads_per_core.value}")

        result = cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device)
        if result == CUDA_SUCCESS:
            print("  GPU clock: %g MHz" % (clockrate.value / 1000.0))
        result = cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device)
        if result == CUDA_SUCCESS:
            print(f"  Memory clock: %g MHz" % (clockrate.value / 1000.0))
        try:
            result = cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device)
        except AttributeError:
            result = cuda.cuCtxCreate(ctypes.byref(context), 0, device)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            print("cuCtxCreate failed with error code %d: %s" % (result, error_str.value.decode()))
        else:
            try:
                result = cuda.cuMemGetInfo_v2(ctypes.byref(freeMem), ctypes.byref(totalMem))
            except AttributeError:
                result = cuda.cuMemGetInfo(ctypes.byref(freeMem), ctypes.byref(totalMem))
            if result == CUDA_SUCCESS:
                print("  Total Memory: %ld MiB" % (totalMem.value / 1024**2))
                print("  Free Memory: %ld MiB" % (freeMem.value / 1024**2))
            else:
                cuda.cuGetErrorString(result, ctypes.byref(error_str))
                print("cuMemGetInfo failed with error code %d: %s" % (result, error_str.value.decode()))
            cuda.cuCtxDetach(context)


if __name__ == "__main__":
    main()
