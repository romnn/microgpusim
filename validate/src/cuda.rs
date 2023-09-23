use color_eyre::eyre;
use rustacuda::error::{CudaError, CudaResult};

pub trait ToResult {
    fn to_result(self) -> CudaResult<()>;
}

impl ToResult for cuda_driver_sys::cudaError_enum {
    fn to_result(self) -> rustacuda::error::CudaResult<()> {
        match self {
            Self::CUDA_SUCCESS => Ok(()),
            Self::CUDA_ERROR_INVALID_VALUE => Err(CudaError::InvalidValue),
            Self::CUDA_ERROR_OUT_OF_MEMORY => Err(CudaError::OutOfMemory),
            Self::CUDA_ERROR_NOT_INITIALIZED => Err(CudaError::NotInitialized),
            Self::CUDA_ERROR_DEINITIALIZED => Err(CudaError::Deinitialized),
            Self::CUDA_ERROR_PROFILER_DISABLED => Err(CudaError::ProfilerDisabled),
            Self::CUDA_ERROR_PROFILER_NOT_INITIALIZED => Err(CudaError::ProfilerNotInitialized),
            Self::CUDA_ERROR_PROFILER_ALREADY_STARTED => Err(CudaError::ProfilerAlreadyStarted),
            Self::CUDA_ERROR_PROFILER_ALREADY_STOPPED => Err(CudaError::ProfilerAlreadyStopped),
            Self::CUDA_ERROR_NO_DEVICE => Err(CudaError::NoDevice),
            Self::CUDA_ERROR_INVALID_DEVICE => Err(CudaError::InvalidDevice),
            Self::CUDA_ERROR_INVALID_IMAGE => Err(CudaError::InvalidImage),
            Self::CUDA_ERROR_INVALID_CONTEXT => Err(CudaError::InvalidContext),
            Self::CUDA_ERROR_CONTEXT_ALREADY_CURRENT => Err(CudaError::ContextAlreadyCurrent),
            Self::CUDA_ERROR_MAP_FAILED => Err(CudaError::MapFailed),
            Self::CUDA_ERROR_UNMAP_FAILED => Err(CudaError::UnmapFailed),
            Self::CUDA_ERROR_ARRAY_IS_MAPPED => Err(CudaError::ArrayIsMapped),
            Self::CUDA_ERROR_ALREADY_MAPPED => Err(CudaError::AlreadyMapped),
            Self::CUDA_ERROR_NO_BINARY_FOR_GPU => Err(CudaError::NoBinaryForGpu),
            Self::CUDA_ERROR_ALREADY_ACQUIRED => Err(CudaError::AlreadyAcquired),
            Self::CUDA_ERROR_NOT_MAPPED => Err(CudaError::NotMapped),
            Self::CUDA_ERROR_NOT_MAPPED_AS_ARRAY => Err(CudaError::NotMappedAsArray),
            Self::CUDA_ERROR_NOT_MAPPED_AS_POINTER => Err(CudaError::NotMappedAsPointer),
            Self::CUDA_ERROR_ECC_UNCORRECTABLE => Err(CudaError::EccUncorrectable),
            Self::CUDA_ERROR_UNSUPPORTED_LIMIT => Err(CudaError::UnsupportedLimit),
            Self::CUDA_ERROR_CONTEXT_ALREADY_IN_USE => Err(CudaError::ContextAlreadyInUse),
            Self::CUDA_ERROR_PEER_ACCESS_UNSUPPORTED => Err(CudaError::PeerAccessUnsupported),
            Self::CUDA_ERROR_INVALID_PTX => Err(CudaError::InvalidPtx),
            Self::CUDA_ERROR_INVALID_GRAPHICS_CONTEXT => Err(CudaError::InvalidGraphicsContext),
            Self::CUDA_ERROR_NVLINK_UNCORRECTABLE => Err(CudaError::NvlinkUncorrectable),
            Self::CUDA_ERROR_INVALID_SOURCE => Err(CudaError::InvalidSouce),
            Self::CUDA_ERROR_FILE_NOT_FOUND => Err(CudaError::FileNotFound),
            Self::CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND => {
                Err(CudaError::SharedObjectSymbolNotFound)
            }
            Self::CUDA_ERROR_SHARED_OBJECT_INIT_FAILED => Err(CudaError::SharedObjectInitFailed),
            Self::CUDA_ERROR_OPERATING_SYSTEM => Err(CudaError::OperatingSystemError),
            Self::CUDA_ERROR_INVALID_HANDLE => Err(CudaError::InvalidHandle),
            Self::CUDA_ERROR_NOT_FOUND => Err(CudaError::NotFound),
            Self::CUDA_ERROR_NOT_READY => Err(CudaError::NotReady),
            Self::CUDA_ERROR_ILLEGAL_ADDRESS => Err(CudaError::IllegalAddress),
            Self::CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES => Err(CudaError::LaunchOutOfResources),
            Self::CUDA_ERROR_LAUNCH_TIMEOUT => Err(CudaError::LaunchTimeout),
            Self::CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING => {
                Err(CudaError::LaunchIncompatibleTexturing)
            }
            Self::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => {
                Err(CudaError::PeerAccessAlreadyEnabled)
            }
            Self::CUDA_ERROR_PEER_ACCESS_NOT_ENABLED => Err(CudaError::PeerAccessNotEnabled),
            Self::CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE => Err(CudaError::PrimaryContextActive),
            Self::CUDA_ERROR_CONTEXT_IS_DESTROYED => Err(CudaError::ContextIsDestroyed),
            Self::CUDA_ERROR_ASSERT => Err(CudaError::AssertError),
            Self::CUDA_ERROR_TOO_MANY_PEERS => Err(CudaError::TooManyPeers),
            Self::CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED => {
                Err(CudaError::HostMemoryAlreadyRegistered)
            }
            Self::CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED => Err(CudaError::HostMemoryNotRegistered),
            Self::CUDA_ERROR_HARDWARE_STACK_ERROR => Err(CudaError::HardwareStackError),
            Self::CUDA_ERROR_ILLEGAL_INSTRUCTION => Err(CudaError::IllegalInstruction),
            Self::CUDA_ERROR_MISALIGNED_ADDRESS => Err(CudaError::MisalignedAddress),
            Self::CUDA_ERROR_INVALID_ADDRESS_SPACE => Err(CudaError::InvalidAddressSpace),
            Self::CUDA_ERROR_INVALID_PC => Err(CudaError::InvalidProgramCounter),
            Self::CUDA_ERROR_LAUNCH_FAILED => Err(CudaError::LaunchFailed),
            Self::CUDA_ERROR_NOT_PERMITTED => Err(CudaError::NotPermitted),
            Self::CUDA_ERROR_NOT_SUPPORTED => Err(CudaError::NotSupported),
            _ => Err(CudaError::UnknownError),
        }
    }
}

/// Flush L2 cache of device.
///
/// based on NVIDIA's `nvbench` [source](https://github.com/NVIDIA/nvbench/blob/f57aa9c993f4392a76650bc54513f571cd1128c9/nvbench/detail/l2flush.cuh#L55).
pub fn flush_l2(device_id: Option<u32>) -> eyre::Result<()> {
    use rustacuda::context::{Context, ContextFlags, CurrentContext};
    use rustacuda::device::{Device, DeviceAttribute};

    rustacuda::init(rustacuda::CudaFlags::empty())?;

    // initialize context for device
    let device = Device::get_device(device_id.unwrap_or(0))?;
    let context =
        Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
    CurrentContext::set_current(&context)?;

    dbg!(device.get_attribute(DeviceAttribute::GlobalL1CacheSupported)? != 0);

    let l2_size = usize::try_from(device.get_attribute(DeviceAttribute::L2CacheSize)?)?;
    if l2_size == 0 {
        return Ok(());
    }

    let global_mem_size = device.total_memory()?;

    log::info!(
        "invalidating L2 of GPU device {:?} ({})",
        device.uuid()?,
        device.name()?
    );

    // allocate device memory
    // let buffer_size = (10 * l2_size).min(global_mem_size);
    let buffer_size = (global_mem_size / 2).max(l2_size);
    dbg!(buffer_size);
    assert_eq!(
        buffer_size,
        buffer_size
            .checked_mul(std::mem::size_of::<std::ffi::c_char>())
            .unwrap_or(0)
    );
    let device_buffer = unsafe { rustacuda::memory::cuda_malloc::<std::ffi::c_char>(buffer_size)? };

    dbg!(human_bytes::human_bytes(global_mem_size as f64));
    dbg!(human_bytes::human_bytes(buffer_size as f64));
    dbg!(human_bytes::human_bytes(l2_size as f64));

    // copy chunks of memory
    let chunk_size = l2_size / 2;

    let chunks = (0..buffer_size - chunk_size).step_by(chunk_size);
    for chunk_offset in chunks {
        dbg!(chunk_offset);
        unsafe {
            let chunk: Box<[std::ffi::c_char]> = utils::box_slice![0; chunk_size];
            cuda_driver_sys::cuMemcpyHtoD_v2(
                device_buffer.as_raw() as u64 + chunk_offset as u64,
                chunk.as_ptr() as *const std::ffi::c_char as *mut std::ffi::c_void,
                chunk_size,
            )
            .to_result()?
        }
    }
    // zero out memory
    // let memset_result = unsafe {
    //     cuda_driver_sys::cuMemsetD8_v2(device_buffer.as_raw() as u64, 0, l2_size).to_result()
    // };

    // synchronize
    CurrentContext::synchronize()?;

    // free memory
    let free_result = unsafe { rustacuda::memory::cuda_free(device_buffer) };

    // memset_result?;
    free_result?;

    // int dev_id{};
    // NVBENCH_CUDA_CALL(cudaGetDevice(&dev_id));
    // NVBENCH_CUDA_CALL(cudaDeviceGetAttribute(&m_l2_size, cudaDevAttrL2CacheSize, dev_id));
    // if (m_l2_size > 0)
    // {
    //   void *buffer = m_l2_buffer;
    //   NVBENCH_CUDA_CALL(cudaMalloc(&buffer, m_l2_size));
    //   m_l2_buffer = reinterpret_cast<int *>(buffer);
    // }
    // NVBENCH_CUDA_CALL_NOEXCEPT(cudaFree(m_l2_buffer));
    // if (m_l2_size > 0)
    // NVBENCH_CUDA_CALL(cudaMemsetAsync(m_l2_buffer, 0, m_l2_size, stream));
    Ok(())
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;

    #[test]
    fn test_flush_l2() -> eyre::Result<()> {
        super::flush_l2(None)?;
        Ok(())
    }
}
