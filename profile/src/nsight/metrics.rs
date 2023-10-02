use crate::Metric;

#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Float(pub f32);

impl From<f32> for Float {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

impl From<Float> for f32 {
    fn from(value: Float) -> Self {
        value.0
    }
}

#[derive(thiserror::Error, Debug, PartialEq)]
pub enum ParseFloatError {
    #[error(transparent)]
    Parse(#[from] std::num::ParseFloatError),
    #[error("bad format: {value:?} ({reason})")]
    BadFormat { value: String, reason: String },
}

impl std::str::FromStr for Float {
    type Err = ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut s = s.to_ascii_lowercase();

        if s.chars().count() > 0 {
            let first_comma = s.chars().position(|c| c == ',');
            let first_dot = s.chars().position(|c| c == '.');
            match (first_comma, first_dot) {
                (Some(_), None) => {
                    let num_commas = s.chars().filter(|&c| c == ',').count();
                    if num_commas > 1 {
                        // remove commas
                        s = s
                            .chars()
                            .filter(|&c| c != ',' && c != ' ')
                            .collect::<String>();
                    } else {
                        return Err(ParseFloatError::BadFormat {
                            value: s,
                            reason: "comma without floating point".to_string(),
                        });
                    }
                }
                (Some(first_comma), Some(first_dot)) => {
                    // sanity check comma before dot
                    if first_comma >= first_dot {
                        return Err(ParseFloatError::BadFormat {
                            value: s,
                            reason: "decimal point followed by comma separator".to_string(),
                        });
                    }

                    // todo: sanity check only single dot
                    // remove commas
                    s = s
                        .chars()
                        .filter(|&c| c != ',' && c != ' ')
                        .collect::<String>();
                }
                _ => {}
            }
        }
        let value = f32::from_str(&s)?;
        Ok(Self(value))
    }
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Profiler {
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.avg")]
    pub profiler_replayer_bytes_mem_accessible_avg: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.max")]
    pub profiler_replayer_bytes_mem_accessible_max: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.min")]
    pub profiler_replayer_bytes_mem_accessible_min: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.sum")]
    pub profiler_replayer_bytes_mem_accessible_sum: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.avg")]
    pub profiler_replayer_bytes_mem_backed_up_avg: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.max")]
    pub profiler_replayer_bytes_mem_backed_up_max: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.min")]
    pub profiler_replayer_bytes_mem_backed_up_min: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.sum")]
    pub profiler_replayer_bytes_mem_backed_up_sum: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_passes")]
    pub profiler_replayer_passes: Option<Metric<Float>>,
    #[serde(rename = "profiler__replayer_passes_type_warmup")]
    pub profiler_replayer_passes_type_warmup: Option<Metric<Float>>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Device {
    #[serde(rename = "device__attribute_architecture")]
    pub attribute_architecture: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_async_engine_count")]
    pub attribute_async_engine_count: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_flush_remote_writes")]
    pub attribute_can_flush_remote_writes: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_map_host_memory")]
    pub attribute_can_map_host_memory: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_tex2d_gather")]
    pub attribute_can_tex2d_gather: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_use_64_bit_stream_mem_ops")]
    pub attribute_can_use_64_bit_stream_mem_ops: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_use_host_pointer_for_registered_mem")]
    pub attribute_can_use_host_pointer_for_registered_mem: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_use_stream_mem_ops")]
    pub attribute_can_use_stream_mem_ops: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_can_use_stream_wait_value_nor")]
    pub attribute_can_use_stream_wait_value_nor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_chip")]
    pub attribute_chip: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_clock_rate")]
    pub attribute_clock_rate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_compute_capability_major")]
    pub attribute_compute_capability_major: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_compute_capability_minor")]
    pub attribute_compute_capability_minor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_compute_mode")]
    pub attribute_compute_mode: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_compute_preemption_supported")]
    pub attribute_compute_preemption_supported: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_concurrent_kernels")]
    pub attribute_concurrent_kernels: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_concurrent_managed_access")]
    pub attribute_concurrent_managed_access: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_cooperative_launch")]
    pub attribute_cooperative_launch: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_cooperative_multi_device_launch")]
    pub attribute_cooperative_multi_device_launch: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_device_index")]
    pub attribute_device_index: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_direct_managed_mem_access_from_host")]
    pub attribute_direct_managed_mem_access_from_host: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_display_name")]
    pub attribute_display_name: Option<Metric<String>>,
    #[serde(rename = "device__attribute_ecc_enabled")]
    pub attribute_ecc_enabled: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_fb_bus_width")]
    pub attribute_fb_bus_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_fbp_count")]
    pub attribute_fbp_count: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_global_l1_cache_supported")]
    pub attribute_global_l1_cache_supported: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_global_memory_bus_width")]
    pub attribute_global_memory_bus_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_overlap")]
    pub attribute_gpu_overlap: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_device_id")]
    pub attribute_gpu_pci_device_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_device_id")]
    pub attribute_gpu_pci_ext_device_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_downstream_link_rate")]
    pub attribute_gpu_pci_ext_downstream_link_rate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_downstream_link_width")]
    pub attribute_gpu_pci_ext_downstream_link_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gen")]
    pub attribute_gpu_pci_ext_gen: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gpu_gen")]
    pub attribute_gpu_pci_ext_gpu_gen: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gpu_link_rate")]
    pub attribute_gpu_pci_ext_gpu_link_rate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gpu_link_width")]
    pub attribute_gpu_pci_ext_gpu_link_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_revision_id")]
    pub attribute_gpu_pci_revision_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_gpu_pci_sub_system_id")]
    pub attribute_gpu_pci_sub_system_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_host_native_atomic_supported")]
    pub attribute_host_native_atomic_supported: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_host_register_supported")]
    pub attribute_host_register_supported: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_implementation")]
    pub attribute_implementation: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_integrated")]
    pub attribute_integrated: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_kernel_exec_timeout")]
    pub attribute_kernel_exec_timeout: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_l2_cache_size")]
    pub attribute_l2_cache_size: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_l2s_count")]
    pub attribute_l2s_count: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_limits_max_cta_per_sm")]
    pub attribute_limits_max_cta_per_sm: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_local_l1_cache_supported")]
    pub attribute_local_l1_cache_supported: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_managed_memory")]
    pub attribute_managed_memory: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_block_dim_x")]
    pub attribute_max_block_dim_x: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_block_dim_y")]
    pub attribute_max_block_dim_y: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_block_dim_z")]
    pub attribute_max_block_dim_z: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_gpu_frequency_khz")]
    pub attribute_max_gpu_frequency_khz: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_grid_dim_x")]
    pub attribute_max_grid_dim_x: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_grid_dim_y")]
    pub attribute_max_grid_dim_y: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_grid_dim_z")]
    pub attribute_max_grid_dim_z: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_ipc_per_multiprocessor")]
    pub attribute_max_ipc_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_ipc_per_scheduler")]
    pub attribute_max_ipc_per_scheduler: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_mem_frequency_khz")]
    pub attribute_max_mem_frequency_khz: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_pitch")]
    pub attribute_max_pitch: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_registers_per_block")]
    pub attribute_max_registers_per_block: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_registers_per_multiprocessor")]
    pub attribute_max_registers_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_registers_per_thread")]
    pub attribute_max_registers_per_thread: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_shared_memory_per_block")]
    pub attribute_max_shared_memory_per_block: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_shared_memory_per_block_optin")]
    pub attribute_max_shared_memory_per_block_optin: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_shared_memory_per_multiprocessor")]
    pub attribute_max_shared_memory_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_threads_per_block")]
    pub attribute_max_threads_per_block: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_threads_per_multiprocessor")]
    pub attribute_max_threads_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_warps_per_multiprocessor")]
    pub attribute_max_warps_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_max_warps_per_scheduler")]
    pub attribute_max_warps_per_scheduler: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface1d_layered_layers")]
    pub attribute_maximum_surface1d_layered_layers: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface1d_layered_width")]
    pub attribute_maximum_surface1d_layered_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface1d_width")]
    pub attribute_maximum_surface1d_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface2d_height")]
    pub attribute_maximum_surface2d_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface2d_layered_height")]
    pub attribute_maximum_surface2d_layered_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface2d_layered_layers")]
    pub attribute_maximum_surface2d_layered_layers: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface2d_layered_width")]
    pub attribute_maximum_surface2d_layered_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface2d_width")]
    pub attribute_maximum_surface2d_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface3d_depth")]
    pub attribute_maximum_surface3d_depth: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface3d_height")]
    pub attribute_maximum_surface3d_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surface3d_width")]
    pub attribute_maximum_surface3d_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surfacecubemap_layered_layers")]
    pub attribute_maximum_surfacecubemap_layered_layers: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surfacecubemap_layered_width")]
    pub attribute_maximum_surfacecubemap_layered_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_surfacecubemap_width")]
    pub attribute_maximum_surfacecubemap_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture1d_layered_layers")]
    pub attribute_maximum_texture1d_layered_layers: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture1d_layered_width")]
    pub attribute_maximum_texture1d_layered_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture1d_linear_width")]
    pub attribute_maximum_texture1d_linear_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture1d_mipmapped_width")]
    pub attribute_maximum_texture1d_mipmapped_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture1d_width")]
    pub attribute_maximum_texture1d_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_gather_height")]
    pub attribute_maximum_texture2d_gather_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_gather_width")]
    pub attribute_maximum_texture2d_gather_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_height")]
    pub attribute_maximum_texture2d_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_layered_height")]
    pub attribute_maximum_texture2d_layered_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_layered_layers")]
    pub attribute_maximum_texture2d_layered_layers: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_layered_width")]
    pub attribute_maximum_texture2d_layered_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_linear_height")]
    pub attribute_maximum_texture2d_linear_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_linear_pitch")]
    pub attribute_maximum_texture2d_linear_pitch: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_linear_width")]
    pub attribute_maximum_texture2d_linear_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_mipmapped_height")]
    pub attribute_maximum_texture2d_mipmapped_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_mipmapped_width")]
    pub attribute_maximum_texture2d_mipmapped_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture2d_width")]
    pub attribute_maximum_texture2d_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture3d_depth")]
    pub attribute_maximum_texture3d_depth: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture3d_depth_alternate")]
    pub attribute_maximum_texture3d_depth_alternate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture3d_height")]
    pub attribute_maximum_texture3d_height: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture3d_height_alternate")]
    pub attribute_maximum_texture3d_height_alternate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture3d_width")]
    pub attribute_maximum_texture3d_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texture3d_width_alternate")]
    pub attribute_maximum_texture3d_width_alternate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texturecubemap_layered_layers")]
    pub attribute_maximum_texturecubemap_layered_layers: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texturecubemap_layered_width")]
    pub attribute_maximum_texturecubemap_layered_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_maximum_texturecubemap_width")]
    pub attribute_maximum_texturecubemap_width: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_memory_clock_rate")]
    pub attribute_memory_clock_rate: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_multi_gpu_board")]
    pub attribute_multi_gpu_board: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_multi_gpu_board_group_id")]
    pub attribute_multi_gpu_board_group_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_multiprocessor_count")]
    pub attribute_multiprocessor_count: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_num_l2s_per_fbp")]
    pub attribute_num_l2s_per_fbp: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_num_schedulers_per_multiprocessor")]
    pub attribute_num_schedulers_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_num_tex_per_multiprocessor")]
    pub attribute_num_tex_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_pageable_memory_access")]
    pub attribute_pageable_memory_access: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_pageable_memory_access_uses_host_page_tables")]
    pub attribute_pageable_memory_access_uses_host_page_tables: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_pci_bus_id")]
    pub attribute_pci_bus_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_pci_device_id")]
    pub attribute_pci_device_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_pci_domain_id")]
    pub attribute_pci_domain_id: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_ram_location")]
    pub attribute_ram_location: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_ram_type")]
    pub attribute_ram_type: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_sass_level")]
    pub attribute_sass_level: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_single_to_double_precision_perf_ratio")]
    pub attribute_single_to_double_precision_perf_ratio: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_stream_priorities_supported")]
    pub attribute_stream_priorities_supported: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_surface_alignment")]
    pub attribute_surface_alignment: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_tcc_driver")]
    pub attribute_tcc_driver: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_texture_alignment")]
    pub attribute_texture_alignment: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_texture_pitch_alignment")]
    pub attribute_texture_pitch_alignment: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_total_constant_memory")]
    pub attribute_total_constant_memory: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_total_memory")]
    pub attribute_total_memory: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_unified_addressing")]
    pub attribute_unified_addressing: Option<Metric<Float>>,
    #[serde(rename = "device__attribute_warp_size")]
    pub attribute_warp_size: Option<Metric<Float>>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DRAM {
    #[serde(rename = "dram__bytes_per_sec")]
    pub bytes_per_sec: Option<Metric<Float>>,
    #[serde(rename = "dram__frequency")]
    pub frequency: Option<Metric<Float>>,
    #[serde(rename = "dram__read_bytes")]
    pub read_bytes: Option<Metric<Float>>,
    #[serde(rename = "dram__read_bytes_per_sec")]
    pub read_bytes_per_sec: Option<Metric<Float>>,
    #[serde(rename = "dram__read_pct")]
    pub read_pct: Option<Metric<Float>>,
    #[serde(rename = "dram__read_sectors")]
    pub read_sectors: Option<Metric<Float>>,
    #[serde(rename = "dram__write_bytes")]
    pub write_bytes: Option<Metric<Float>>,
    #[serde(rename = "dram__write_bytes_per_sec")]
    pub write_bytes_per_sec: Option<Metric<Float>>,
    #[serde(rename = "dram__write_pct")]
    pub write_pct: Option<Metric<Float>>,
    #[serde(rename = "dram__write_sectors")]
    pub write_sectors: Option<Metric<Float>>,
    #[serde(rename = "dram__sectors_read.sum")]
    pub sectors_read_sum: Option<Metric<Float>>,
    #[serde(rename = "dram__sectors_write.sum")]
    pub sectors_write_sum: Option<Metric<Float>>,
    #[serde(rename = "dram__bytes_read.sum")]
    pub bytes_read_sum: Option<Metric<Float>>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct L1Tex {
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_ld_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_st_lookup_hit_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_st_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss_sum: Option<Metric<Float>>,
    #[serde(rename = "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum")]
    pub l1tex_t_sectors_pipe_lsu_mem_global_op_st_lookup_miss_sum: Option<Metric<Float>>,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    #[serde(rename = "ID")]
    pub id: Option<Metric<usize>>,
    #[serde(rename = "Process ID")]
    pub process_id: Option<Metric<usize>>,
    #[serde(rename = "Process Name")]
    pub process_name: Option<Metric<String>>,
    #[serde(rename = "Host Name")]
    pub host_name: Option<Metric<String>>,
    #[serde(rename = "Kernel Name")]
    pub kernel_name: Option<Metric<String>>,
    #[serde(rename = "Kernel Time")]
    pub kernel_time: Option<Metric<String>>,
    #[serde(rename = "Context")]
    pub context: Option<Metric<usize>>,
    #[serde(rename = "Stream")]
    pub stream: Option<Metric<usize>>,

    #[serde(flatten)]
    pub device: Device,
    #[serde(flatten)]
    pub dram: DRAM,
    #[serde(flatten)]
    pub profiler: Profiler,

    #[serde(rename = "fbpa__sol_pct")]
    pub fbpa_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "gpc__elapsed_cycles_max")]
    pub gpc_elapsed_cycles_max: Option<Metric<Float>>,
    #[serde(rename = "gpc__elapsed_cycles.avg")]
    pub gpc_elapsed_cycles_avg: Option<Metric<Float>>,
    #[serde(rename = "gpc__frequency")]
    pub gpc_frequency: Option<Metric<Float>>,
    #[serde(rename = "gpu__compute_memory_request_utilization_pct")]
    pub gpu_compute_memory_request_utilization_pct: Option<Metric<Float>>,
    #[serde(rename = "gpu__compute_memory_sol_pct")]
    pub gpu_compute_memory_sol_pct: Option<Metric<Float>>,
    #[serde(rename = "gpu__time_duration")]
    pub gpu_time_duration: Option<Metric<Float>>,
    #[serde(rename = "inst_executed")]
    pub inst_executed: Option<Metric<String>>,
    #[serde(rename = "launch__block_size")]
    pub launch_block_size: Option<Metric<Float>>,
    #[serde(rename = "launch__context_id")]
    pub launch_context_id: Option<Metric<Float>>,
    #[serde(rename = "launch__function_pcs")]
    pub launch_function_pcs: Option<Metric<Float>>,
    #[serde(rename = "launch__grid_size")]
    pub launch_grid_size: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_blocks")]
    pub launch_occupancy_limit_blocks: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_registers")]
    pub launch_occupancy_limit_registers: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_shared_mem")]
    pub launch_occupancy_limit_shared_mem: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_limit_warps")]
    pub launch_occupancy_limit_warps: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_per_block_size")]
    pub launch_occupancy_per_block_size: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_per_register_count")]
    pub launch_occupancy_per_register_count: Option<Metric<Float>>,
    #[serde(rename = "launch__occupancy_per_shared_mem_size")]
    pub launch_occupancy_per_shared_mem_size: Option<Metric<Float>>,
    #[serde(rename = "launch__registers_per_thread")]
    pub launch_registers_per_thread: Option<Metric<Float>>,
    #[serde(rename = "launch__registers_per_thread_allocated")]
    pub launch_registers_per_thread_allocated: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_config_size")]
    pub launch_shared_mem_config_size: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_per_block_allocated")]
    pub launch_shared_mem_per_block_allocated: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_per_block_dynamic")]
    pub launch_shared_mem_per_block_dynamic: Option<Metric<Float>>,
    #[serde(rename = "launch__shared_mem_per_block_static")]
    pub launch_shared_mem_per_block_static: Option<Metric<Float>>,
    #[serde(rename = "launch__stream_id")]
    pub launch_stream_id: Option<Metric<Float>>,
    #[serde(rename = "launch__thread_count")]
    pub launch_thread_count: Option<Metric<Float>>,
    #[serde(rename = "launch__waves_per_multiprocessor")]
    pub launch_waves_per_multiprocessor: Option<Metric<Float>>,
    #[serde(rename = "ltc__sol_pct")]
    pub ltc_sol_pct: Option<Metric<Float>>,

    #[serde(flatten)]
    pub lts: super::lts::LTS,

    #[serde(rename = "memory_access_size_type")]
    pub memory_access_size_type: Option<Metric<String>>,
    #[serde(rename = "memory_access_type")]
    pub memory_access_type: Option<Metric<String>>,
    #[serde(rename = "memory_l2_transactions_global")]
    pub memory_l2_transactions_global: Option<Metric<String>>,
    #[serde(rename = "memory_l2_transactions_local")]
    pub memory_l2_transactions_local: Option<Metric<String>>,
    #[serde(rename = "memory_shared_transactions")]
    pub memory_shared_transactions: Option<Metric<Float>>,
    #[serde(rename = "memory_type")]
    pub memory_type: Option<Metric<String>>,

    #[serde(rename = "sass__block_histogram")]
    pub sass_block_histogram: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_global_loads")]
    pub sass_inst_executed_global_loads: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_global_stores")]
    pub sass_inst_executed_global_stores: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_local_loads")]
    pub sass_inst_executed_local_loads: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_local_stores")]
    pub sass_inst_executed_local_stores: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_per_opcode")]
    pub sass_inst_executed_per_opcode: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_shared_loads")]
    pub sass_inst_executed_shared_loads: Option<Metric<Float>>,
    #[serde(rename = "sass__inst_executed_shared_stores")]
    pub sass_inst_executed_shared_stores: Option<Metric<Float>>,
    #[serde(rename = "sass__warp_histogram")]
    pub sass_warp_histogram: Option<Metric<Float>>,

    #[serde(flatten)]
    pub sm: super::sm::SM,
    #[serde(flatten)]
    pub sm_scheduler: super::scheduler::SMScheduler,
    #[serde(flatten)]
    pub tex: super::tex::Tex,

    #[serde(rename = "thread_inst_executed_true")]
    pub thread_inst_executed_true: Option<Metric<String>>,

    #[serde(flatten)]
    pub other: std::collections::HashMap<String, Metric<serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::Float;
    use std::str::FromStr;

    #[test]
    fn test_parse_float() {
        assert_eq!(Float::from_str("12"), Ok(12.0.into()));
        assert_eq!(
            Float::from_str("12,00").ok(),
            None,
            "cannot interpret single comma"
        );
        assert_eq!(
            Float::from_str("12,001,233"),
            Ok(12_001_233.0.into()),
            "multiple comma separators disambiguate"
        );
        assert_eq!(
            Float::from_str("12,001,233.5347"),
            Ok(12_001_233.5347.into())
        );
        assert_eq!(
            Float::from_str("12.001,233").ok(),
            None,
            "cannot have decimal point followed by comma separator"
        );
        assert_eq!(Float::from_str("-0.5"), Ok((-0.5).into()));
    }
}
