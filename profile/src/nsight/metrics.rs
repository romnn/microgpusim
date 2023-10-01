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

        assert!(s.chars().count() > 0);
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
                    // assert!(first_comma < first_dot);
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
pub struct Metrics {
    #[serde(rename = "ID")]
    pub id: Metric<usize>,
    #[serde(rename = "Process ID")]
    pub process_id: Metric<usize>,
    #[serde(rename = "Process Name")]
    pub process_name: Metric<String>,
    #[serde(rename = "Host Name")]
    pub host_name: Metric<String>,
    #[serde(rename = "Kernel Name")]
    pub kernel_name: Metric<String>,
    #[serde(rename = "Kernel Time")]
    pub kernel_time: Metric<String>,
    #[serde(rename = "Context")]
    pub context: Metric<usize>,
    #[serde(rename = "Stream")]
    pub stream: Metric<usize>,
    #[serde(rename = "device__attribute_architecture")]
    pub device_attribute_architecture: Metric<Float>,
    #[serde(rename = "device__attribute_async_engine_count")]
    pub device_attribute_async_engine_count: Metric<Float>,
    #[serde(rename = "device__attribute_can_flush_remote_writes")]
    pub device_attribute_can_flush_remote_writes: Metric<Float>,
    #[serde(rename = "device__attribute_can_map_host_memory")]
    pub device_attribute_can_map_host_memory: Metric<Float>,
    #[serde(rename = "device__attribute_can_tex2d_gather")]
    pub device_attribute_can_tex2d_gather: Metric<Float>,
    #[serde(rename = "device__attribute_can_use_64_bit_stream_mem_ops")]
    pub device_attribute_can_use_64_bit_stream_mem_ops: Metric<Float>,
    #[serde(rename = "device__attribute_can_use_host_pointer_for_registered_mem")]
    pub device_attribute_can_use_host_pointer_for_registered_mem: Metric<Float>,
    #[serde(rename = "device__attribute_can_use_stream_mem_ops")]
    pub device_attribute_can_use_stream_mem_ops: Metric<Float>,
    #[serde(rename = "device__attribute_can_use_stream_wait_value_nor")]
    pub device_attribute_can_use_stream_wait_value_nor: Metric<Float>,
    #[serde(rename = "device__attribute_chip")]
    pub device_attribute_chip: Metric<Float>,
    #[serde(rename = "device__attribute_clock_rate")]
    pub device_attribute_clock_rate: Metric<Float>,
    #[serde(rename = "device__attribute_compute_capability_major")]
    pub device_attribute_compute_capability_major: Metric<Float>,
    #[serde(rename = "device__attribute_compute_capability_minor")]
    pub device_attribute_compute_capability_minor: Metric<Float>,
    #[serde(rename = "device__attribute_compute_mode")]
    pub device_attribute_compute_mode: Metric<Float>,
    #[serde(rename = "device__attribute_compute_preemption_supported")]
    pub device_attribute_compute_preemption_supported: Metric<Float>,
    #[serde(rename = "device__attribute_concurrent_kernels")]
    pub device_attribute_concurrent_kernels: Metric<Float>,
    #[serde(rename = "device__attribute_concurrent_managed_access")]
    pub device_attribute_concurrent_managed_access: Metric<Float>,
    #[serde(rename = "device__attribute_cooperative_launch")]
    pub device_attribute_cooperative_launch: Metric<Float>,
    #[serde(rename = "device__attribute_cooperative_multi_device_launch")]
    pub device_attribute_cooperative_multi_device_launch: Metric<Float>,
    #[serde(rename = "device__attribute_device_index")]
    pub device_attribute_device_index: Metric<Float>,
    #[serde(rename = "device__attribute_direct_managed_mem_access_from_host")]
    pub device_attribute_direct_managed_mem_access_from_host: Metric<Float>,
    #[serde(rename = "device__attribute_display_name")]
    pub device_attribute_display_name: Metric<String>,
    #[serde(rename = "device__attribute_ecc_enabled")]
    pub device_attribute_ecc_enabled: Metric<Float>,
    #[serde(rename = "device__attribute_fb_bus_width")]
    pub device_attribute_fb_bus_width: Metric<Float>,
    #[serde(rename = "device__attribute_fbp_count")]
    pub device_attribute_fbp_count: Metric<Float>,
    #[serde(rename = "device__attribute_global_l1_cache_supported")]
    pub device_attribute_global_l1_cache_supported: Metric<Float>,
    #[serde(rename = "device__attribute_global_memory_bus_width")]
    pub device_attribute_global_memory_bus_width: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_overlap")]
    pub device_attribute_gpu_overlap: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_device_id")]
    pub device_attribute_gpu_pci_device_id: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_device_id")]
    pub device_attribute_gpu_pci_ext_device_id: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_downstream_link_rate")]
    pub device_attribute_gpu_pci_ext_downstream_link_rate: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_downstream_link_width")]
    pub device_attribute_gpu_pci_ext_downstream_link_width: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gen")]
    pub device_attribute_gpu_pci_ext_gen: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gpu_gen")]
    pub device_attribute_gpu_pci_ext_gpu_gen: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gpu_link_rate")]
    pub device_attribute_gpu_pci_ext_gpu_link_rate: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_ext_gpu_link_width")]
    pub device_attribute_gpu_pci_ext_gpu_link_width: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_revision_id")]
    pub device_attribute_gpu_pci_revision_id: Metric<Float>,
    #[serde(rename = "device__attribute_gpu_pci_sub_system_id")]
    pub device_attribute_gpu_pci_sub_system_id: Metric<Float>,
    #[serde(rename = "device__attribute_host_native_atomic_supported")]
    pub device_attribute_host_native_atomic_supported: Metric<Float>,
    #[serde(rename = "device__attribute_host_register_supported")]
    pub device_attribute_host_register_supported: Metric<Float>,
    #[serde(rename = "device__attribute_implementation")]
    pub device_attribute_implementation: Metric<Float>,
    #[serde(rename = "device__attribute_integrated")]
    pub device_attribute_integrated: Metric<Float>,
    #[serde(rename = "device__attribute_kernel_exec_timeout")]
    pub device_attribute_kernel_exec_timeout: Metric<Float>,
    #[serde(rename = "device__attribute_l2_cache_size")]
    pub device_attribute_l2_cache_size: Metric<Float>,
    #[serde(rename = "device__attribute_l2s_count")]
    pub device_attribute_l2s_count: Metric<Float>,
    #[serde(rename = "device__attribute_limits_max_cta_per_sm")]
    pub device_attribute_limits_max_cta_per_sm: Metric<Float>,
    #[serde(rename = "device__attribute_local_l1_cache_supported")]
    pub device_attribute_local_l1_cache_supported: Metric<Float>,
    #[serde(rename = "device__attribute_managed_memory")]
    pub device_attribute_managed_memory: Metric<Float>,
    #[serde(rename = "device__attribute_max_block_dim_x")]
    pub device_attribute_max_block_dim_x: Metric<Float>,
    #[serde(rename = "device__attribute_max_block_dim_y")]
    pub device_attribute_max_block_dim_y: Metric<Float>,
    #[serde(rename = "device__attribute_max_block_dim_z")]
    pub device_attribute_max_block_dim_z: Metric<Float>,
    #[serde(rename = "device__attribute_max_gpu_frequency_khz")]
    pub device_attribute_max_gpu_frequency_khz: Metric<Float>,
    #[serde(rename = "device__attribute_max_grid_dim_x")]
    pub device_attribute_max_grid_dim_x: Metric<Float>,
    #[serde(rename = "device__attribute_max_grid_dim_y")]
    pub device_attribute_max_grid_dim_y: Metric<Float>,
    #[serde(rename = "device__attribute_max_grid_dim_z")]
    pub device_attribute_max_grid_dim_z: Metric<Float>,
    #[serde(rename = "device__attribute_max_ipc_per_multiprocessor")]
    pub device_attribute_max_ipc_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_max_ipc_per_scheduler")]
    pub device_attribute_max_ipc_per_scheduler: Metric<Float>,
    #[serde(rename = "device__attribute_max_mem_frequency_khz")]
    pub device_attribute_max_mem_frequency_khz: Metric<Float>,
    #[serde(rename = "device__attribute_max_pitch")]
    pub device_attribute_max_pitch: Metric<Float>,
    #[serde(rename = "device__attribute_max_registers_per_block")]
    pub device_attribute_max_registers_per_block: Metric<Float>,
    #[serde(rename = "device__attribute_max_registers_per_multiprocessor")]
    pub device_attribute_max_registers_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_max_registers_per_thread")]
    pub device_attribute_max_registers_per_thread: Metric<Float>,
    #[serde(rename = "device__attribute_max_shared_memory_per_block")]
    pub device_attribute_max_shared_memory_per_block: Metric<Float>,
    #[serde(rename = "device__attribute_max_shared_memory_per_block_optin")]
    pub device_attribute_max_shared_memory_per_block_optin: Metric<Float>,
    #[serde(rename = "device__attribute_max_shared_memory_per_multiprocessor")]
    pub device_attribute_max_shared_memory_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_max_threads_per_block")]
    pub device_attribute_max_threads_per_block: Metric<Float>,
    #[serde(rename = "device__attribute_max_threads_per_multiprocessor")]
    pub device_attribute_max_threads_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_max_warps_per_multiprocessor")]
    pub device_attribute_max_warps_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_max_warps_per_scheduler")]
    pub device_attribute_max_warps_per_scheduler: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface1d_layered_layers")]
    pub device_attribute_maximum_surface1d_layered_layers: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface1d_layered_width")]
    pub device_attribute_maximum_surface1d_layered_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface1d_width")]
    pub device_attribute_maximum_surface1d_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface2d_height")]
    pub device_attribute_maximum_surface2d_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface2d_layered_height")]
    pub device_attribute_maximum_surface2d_layered_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface2d_layered_layers")]
    pub device_attribute_maximum_surface2d_layered_layers: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface2d_layered_width")]
    pub device_attribute_maximum_surface2d_layered_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface2d_width")]
    pub device_attribute_maximum_surface2d_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface3d_depth")]
    pub device_attribute_maximum_surface3d_depth: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface3d_height")]
    pub device_attribute_maximum_surface3d_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surface3d_width")]
    pub device_attribute_maximum_surface3d_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surfacecubemap_layered_layers")]
    pub device_attribute_maximum_surfacecubemap_layered_layers: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surfacecubemap_layered_width")]
    pub device_attribute_maximum_surfacecubemap_layered_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_surfacecubemap_width")]
    pub device_attribute_maximum_surfacecubemap_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture1d_layered_layers")]
    pub device_attribute_maximum_texture1d_layered_layers: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture1d_layered_width")]
    pub device_attribute_maximum_texture1d_layered_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture1d_linear_width")]
    pub device_attribute_maximum_texture1d_linear_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture1d_mipmapped_width")]
    pub device_attribute_maximum_texture1d_mipmapped_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture1d_width")]
    pub device_attribute_maximum_texture1d_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_gather_height")]
    pub device_attribute_maximum_texture2d_gather_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_gather_width")]
    pub device_attribute_maximum_texture2d_gather_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_height")]
    pub device_attribute_maximum_texture2d_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_layered_height")]
    pub device_attribute_maximum_texture2d_layered_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_layered_layers")]
    pub device_attribute_maximum_texture2d_layered_layers: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_layered_width")]
    pub device_attribute_maximum_texture2d_layered_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_linear_height")]
    pub device_attribute_maximum_texture2d_linear_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_linear_pitch")]
    pub device_attribute_maximum_texture2d_linear_pitch: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_linear_width")]
    pub device_attribute_maximum_texture2d_linear_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_mipmapped_height")]
    pub device_attribute_maximum_texture2d_mipmapped_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_mipmapped_width")]
    pub device_attribute_maximum_texture2d_mipmapped_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture2d_width")]
    pub device_attribute_maximum_texture2d_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture3d_depth")]
    pub device_attribute_maximum_texture3d_depth: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture3d_depth_alternate")]
    pub device_attribute_maximum_texture3d_depth_alternate: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture3d_height")]
    pub device_attribute_maximum_texture3d_height: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture3d_height_alternate")]
    pub device_attribute_maximum_texture3d_height_alternate: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture3d_width")]
    pub device_attribute_maximum_texture3d_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texture3d_width_alternate")]
    pub device_attribute_maximum_texture3d_width_alternate: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texturecubemap_layered_layers")]
    pub device_attribute_maximum_texturecubemap_layered_layers: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texturecubemap_layered_width")]
    pub device_attribute_maximum_texturecubemap_layered_width: Metric<Float>,
    #[serde(rename = "device__attribute_maximum_texturecubemap_width")]
    pub device_attribute_maximum_texturecubemap_width: Metric<Float>,
    #[serde(rename = "device__attribute_memory_clock_rate")]
    pub device_attribute_memory_clock_rate: Metric<Float>,
    #[serde(rename = "device__attribute_multi_gpu_board")]
    pub device_attribute_multi_gpu_board: Metric<Float>,
    #[serde(rename = "device__attribute_multi_gpu_board_group_id")]
    pub device_attribute_multi_gpu_board_group_id: Metric<Float>,
    #[serde(rename = "device__attribute_multiprocessor_count")]
    pub device_attribute_multiprocessor_count: Metric<Float>,
    #[serde(rename = "device__attribute_num_l2s_per_fbp")]
    pub device_attribute_num_l2s_per_fbp: Metric<Float>,
    #[serde(rename = "device__attribute_num_schedulers_per_multiprocessor")]
    pub device_attribute_num_schedulers_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_num_tex_per_multiprocessor")]
    pub device_attribute_num_tex_per_multiprocessor: Metric<Float>,
    #[serde(rename = "device__attribute_pageable_memory_access")]
    pub device_attribute_pageable_memory_access: Metric<Float>,
    #[serde(rename = "device__attribute_pageable_memory_access_uses_host_page_tables")]
    pub device_attribute_pageable_memory_access_uses_host_page_tables: Metric<Float>,
    #[serde(rename = "device__attribute_pci_bus_id")]
    pub device_attribute_pci_bus_id: Metric<Float>,
    #[serde(rename = "device__attribute_pci_device_id")]
    pub device_attribute_pci_device_id: Metric<Float>,
    #[serde(rename = "device__attribute_pci_domain_id")]
    pub device_attribute_pci_domain_id: Metric<Float>,
    #[serde(rename = "device__attribute_ram_location")]
    pub device_attribute_ram_location: Metric<Float>,
    #[serde(rename = "device__attribute_ram_type")]
    pub device_attribute_ram_type: Metric<Float>,
    #[serde(rename = "device__attribute_sass_level")]
    pub device_attribute_sass_level: Metric<Float>,
    #[serde(rename = "device__attribute_single_to_double_precision_perf_ratio")]
    pub device_attribute_single_to_double_precision_perf_ratio: Metric<Float>,
    #[serde(rename = "device__attribute_stream_priorities_supported")]
    pub device_attribute_stream_priorities_supported: Metric<Float>,
    #[serde(rename = "device__attribute_surface_alignment")]
    pub device_attribute_surface_alignment: Metric<Float>,
    #[serde(rename = "device__attribute_tcc_driver")]
    pub device_attribute_tcc_driver: Metric<Float>,
    #[serde(rename = "device__attribute_texture_alignment")]
    pub device_attribute_texture_alignment: Metric<Float>,
    #[serde(rename = "device__attribute_texture_pitch_alignment")]
    pub device_attribute_texture_pitch_alignment: Metric<Float>,
    #[serde(rename = "device__attribute_total_constant_memory")]
    pub device_attribute_total_constant_memory: Metric<Float>,
    #[serde(rename = "device__attribute_total_memory")]
    pub device_attribute_total_memory: Metric<Float>,
    #[serde(rename = "device__attribute_unified_addressing")]
    pub device_attribute_unified_addressing: Metric<Float>,
    #[serde(rename = "device__attribute_warp_size")]
    pub device_attribute_warp_size: Metric<Float>,
    #[serde(rename = "dram__bytes_per_sec")]
    pub dram_bytes_per_sec: Metric<Float>,
    #[serde(rename = "dram__frequency")]
    pub dram_frequency: Metric<Float>,
    #[serde(rename = "dram__read_bytes")]
    pub dram_read_bytes: Metric<Float>,
    #[serde(rename = "dram__read_bytes_per_sec")]
    pub dram_read_bytes_per_sec: Metric<Float>,
    #[serde(rename = "dram__read_pct")]
    pub dram_read_pct: Metric<Float>,
    #[serde(rename = "dram__read_sectors")]
    pub dram_read_sectors: Metric<Float>,
    #[serde(rename = "dram__write_bytes")]
    pub dram_write_bytes: Metric<Float>,
    #[serde(rename = "dram__write_bytes_per_sec")]
    pub dram_write_bytes_per_sec: Metric<Float>,
    #[serde(rename = "dram__write_pct")]
    pub dram_write_pc: Metric<Float>,
    #[serde(rename = "dram__write_sectors")]
    pub dram_write_sectors: Metric<Float>,
    #[serde(rename = "fbpa__sol_pct")]
    pub fbpa_sol_pct: Metric<Float>,
    #[serde(rename = "gpc__elapsed_cycles_max")]
    pub gpc_elapsed_cycles_max: Metric<Float>,
    #[serde(rename = "gpc__frequency")]
    pub gpc_frequency: Metric<Float>,
    #[serde(rename = "gpu__compute_memory_request_utilization_pct")]
    pub gpu_compute_memory_request_utilization_pct: Metric<Float>,
    #[serde(rename = "gpu__compute_memory_sol_pct")]
    pub gpu_compute_memory_sol_pct: Metric<Float>,
    #[serde(rename = "gpu__time_duration")]
    pub gpu_time_duration: Metric<Float>,
    #[serde(rename = "inst_executed")]
    pub inst_executed: Metric<String>,
    #[serde(rename = "launch__block_size")]
    pub launch_block_size: Metric<Float>,
    #[serde(rename = "launch__context_id")]
    pub launch_context_id: Metric<Float>,
    #[serde(rename = "launch__function_pcs")]
    pub launch_function_pcs: Metric<Float>,
    #[serde(rename = "launch__grid_size")]
    pub launch_grid_size: Metric<Float>,
    #[serde(rename = "launch__occupancy_limit_blocks")]
    pub launch_occupancy_limit_blocks: Metric<Float>,
    #[serde(rename = "launch__occupancy_limit_registers")]
    pub launch_occupancy_limit_registers: Metric<Float>,
    #[serde(rename = "launch__occupancy_limit_shared_mem")]
    pub launch_occupancy_limit_shared_mem: Metric<Float>,
    #[serde(rename = "launch__occupancy_limit_warps")]
    pub launch_occupancy_limit_warps: Metric<Float>,
    #[serde(rename = "launch__occupancy_per_block_size")]
    pub launch_occupancy_per_block_size: Metric<Float>,
    #[serde(rename = "launch__occupancy_per_register_count")]
    pub launch_occupancy_per_register_count: Metric<Float>,
    #[serde(rename = "launch__occupancy_per_shared_mem_size")]
    pub launch_occupancy_per_shared_mem_size: Metric<Float>,
    #[serde(rename = "launch__registers_per_thread")]
    pub launch_registers_per_thread: Metric<Float>,
    #[serde(rename = "launch__registers_per_thread_allocated")]
    pub launch_registers_per_thread_allocated: Metric<Float>,
    #[serde(rename = "launch__shared_mem_config_size")]
    pub launch_shared_mem_config_size: Metric<Float>,
    #[serde(rename = "launch__shared_mem_per_block_allocated")]
    pub launch_shared_mem_per_block_allocated: Metric<Float>,
    #[serde(rename = "launch__shared_mem_per_block_dynamic")]
    pub launch_shared_mem_per_block_dynamic: Metric<Float>,
    #[serde(rename = "launch__shared_mem_per_block_static")]
    pub launch_shared_mem_per_block_static: Metric<Float>,
    #[serde(rename = "launch__stream_id")]
    pub launch_stream_id: Metric<Float>,
    #[serde(rename = "launch__thread_count")]
    pub launch_thread_count: Metric<Float>,
    #[serde(rename = "launch__waves_per_multiprocessor")]
    pub launch_waves_per_multiprocessor: Metric<Float>,
    #[serde(rename = "ltc__sol_pct")]
    pub ltc_sol_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_atomic_sectors_global_atom_utilization_pct")]
    pub lts_request_tex_atomic_sectors_global_atom_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_atomic_sectors_surface_atom_utilization_pct")]
    pub lts_request_tex_atomic_sectors_surface_atom_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_cached_utilization_pct")]
    pub lts_request_tex_read_sectors_global_ld_cached_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_read_sectors_global_ld_uncached_utilization_pct")]
    pub lts_request_tex_read_sectors_global_ld_uncached_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_cached_utilization_pct")]
    pub lts_request_tex_read_sectors_local_ld_cached_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_read_sectors_local_ld_uncached_utilization_pct")]
    pub lts_request_tex_read_sectors_local_ld_uncached_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_read_sectors_surface_ld_utilization_pct")]
    pub lts_request_tex_read_sectors_surface_ld_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_write_sectors_global_nonatom_utilization_pct")]
    pub lts_request_tex_write_sectors_global_nonatom_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_write_sectors_global_red_utilization_pct")]
    pub lts_request_tex_write_sectors_global_red_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_write_sectors_local_st_utilization_pct")]
    pub lts_request_tex_write_sectors_local_st_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_write_sectors_surface_nonatom_utilization_pct")]
    pub lts_request_tex_write_sectors_surface_nonatom_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_tex_write_sectors_surface_red_utilization_pct")]
    pub lts_request_tex_write_sectors_surface_red_utilization_pct: Metric<Float>,
    #[serde(rename = "lts__request_total_sectors_hitrate_pct")]
    pub lts_request_total_sectors_hitrate_pct: Metric<Float>,
    #[serde(rename = "memory_access_size_type")]
    pub memory_access_size_type: Metric<String>,
    #[serde(rename = "memory_access_type")]
    pub memory_access_type: Metric<String>,
    #[serde(rename = "memory_l2_transactions_global")]
    pub memory_l2_transactions_global: Metric<String>,
    #[serde(rename = "memory_l2_transactions_local")]
    pub memory_l2_transactions_local: Metric<String>,
    #[serde(rename = "memory_shared_transactions")]
    pub memory_shared_transactions: Metric<Float>,
    #[serde(rename = "memory_type")]
    pub memory_type: Metric<String>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.avg")]
    pub profiler_replayer_bytes_mem_accessible_avg: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.max")]
    pub profiler_replayer_bytes_mem_accessible_max: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.min")]
    pub profiler_replayer_bytes_mem_accessible_min: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_accessible.sum")]
    pub profiler_replayer_bytes_mem_accessible_sum: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.avg")]
    pub profiler_replayer_bytes_mem_backed_up_avg: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.max")]
    pub profiler_replayer_bytes_mem_backed_up_max: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.min")]
    pub profiler_replayer_bytes_mem_backed_up_min: Metric<Float>,
    #[serde(rename = "profiler__replayer_bytes_mem_backed_up.sum")]
    pub profiler_replayer_bytes_mem_backed_up_sum: Metric<Float>,
    #[serde(rename = "profiler__replayer_passes")]
    pub profiler_replayer_passes: Metric<Float>,
    #[serde(rename = "profiler__replayer_passes_type_warmup")]
    pub profiler_replayer_passes_type_warmup: Metric<Float>,
    #[serde(rename = "sass__block_histogram")]
    pub sass_block_histogram: Metric<Float>,
    #[serde(rename = "sass__inst_executed_global_loads")]
    pub sass_inst_executed_global_loads: Metric<Float>,
    #[serde(rename = "sass__inst_executed_global_stores")]
    pub sass_inst_executed_global_stores: Metric<Float>,
    #[serde(rename = "sass__inst_executed_local_loads")]
    pub sass_inst_executed_local_loads: Metric<Float>,
    #[serde(rename = "sass__inst_executed_local_stores")]
    pub sass_inst_executed_local_stores: Metric<Float>,
    #[serde(rename = "sass__inst_executed_per_opcode")]
    pub sass_inst_executed_per_opcode: Metric<Float>,
    #[serde(rename = "sass__inst_executed_shared_loads")]
    pub sass_inst_executed_shared_loads: Metric<Float>,
    #[serde(rename = "sass__inst_executed_shared_stores")]
    pub sass_inst_executed_shared_stores: Metric<Float>,
    #[serde(rename = "sass__warp_histogram")]
    pub sass_warp_histogram: Metric<Float>,
    #[serde(rename = "sm__active_warps_avg_per_active_cycle")]
    pub sm_active_warps_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "sm__active_warps_avg_per_active_cycle_pct")]
    pub sm_active_warps_avg_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "sm__elapsed_cycles_avg")]
    pub sm_elapsed_cycles_avg: Metric<Float>,
    #[serde(rename = "sm__inst_executed_avg_per_active_cycle")]
    pub sm_inst_executed_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "sm__inst_executed_avg_per_elapsed_cycle")]
    pub sm_inst_executed_avg_per_elapsed_cycle: Metric<Float>,
    #[serde(rename = "sm__inst_executed_pipes_mem_per_active_cycle_sol_pct")]
    pub sm_inst_executed_pipes_mem_per_active_cycle_sol_pct: Metric<Float>,
    #[serde(rename = "sm__inst_issued_avg_per_active_cycle")]
    pub sm_inst_issued_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "sm__inst_issued_per_active_cycle_sol_pct")]
    pub sm_inst_issued_per_active_cycle_sol_pct: Metric<Float>,
    #[serde(rename = "sm__maximum_warps_avg_per_active_cycle")]
    pub sm_maximum_warps_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "sm__maximum_warps_per_active_cycle_pct")]
    pub sm_maximum_warps_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "sm__shmem_ld_bank_conflict_sum")]
    pub sm_shmem_ld_bank_conflict_sum: Metric<Float>,
    #[serde(rename = "sm__shmem_ld_count_per_active_cycle_sol_pct")]
    pub sm_shmem_ld_count_per_active_cycle_sol_pct: Metric<Float>,
    #[serde(rename = "sm__shmem_ld_count_sum")]
    pub sm_shmem_ld_count_sum: Metric<Float>,
    #[serde(rename = "sm__shmem_st_bank_conflict_sum")]
    pub sm_shmem_st_bank_conflict_sum: Metric<Float>,
    #[serde(rename = "sm__shmem_st_count_per_active_cycle_sol_pct")]
    pub sm_shmem_st_count_per_active_cycle_sol_pct: Metric<Float>,
    #[serde(rename = "sm__shmem_st_count_sum")]
    pub sm_shmem_st_count_sum: Metric<Float>,
    #[serde(rename = "sm__sol_pct")]
    pub sm_sol_pct: Metric<Float>,
    // ===============================================
    #[serde(rename = "smsp__active_warps_avg_per_active_cycle")]
    pub smsp_active_warps_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "smsp__eligible_warps_avg_per_active_cycle")]
    pub smsp_eligible_warps_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_avg")]
    pub smsp_inst_executed_avg: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_global_atomics_sum")]
    pub smsp_inst_executed_global_atomics_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_global_reductions_sum")]
    pub smsp_inst_executed_global_reductions_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_shared_atomics_sum")]
    pub smsp_inst_executed_shared_atomics_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_sum")]
    pub smsp_inst_executed_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_surface_atomics_sum")]
    pub smsp_inst_executed_surface_atomics_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_surface_loads_sum")]
    pub smsp_inst_executed_surface_loads_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_surface_reductions_sum")]
    pub smsp_inst_executed_surface_reductions_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_surface_stores_sum")]
    pub smsp_inst_executed_surface_stores_sum: Metric<Float>,
    #[serde(rename = "smsp__inst_executed_tex_ops")]
    pub smsp_inst_executed_tex_ops: Metric<Float>,
    #[serde(rename = "smsp__inst_issued0_active_per_active_cycle_pct")]
    pub smsp_inst_issued0_active_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "smsp__inst_issued_avg")]
    pub smsp_inst_issued_avg: Metric<Float>,
    #[serde(rename = "smsp__inst_issued_per_issue_active")]
    pub smsp_inst_issued_per_issue_active: Metric<Float>,
    #[serde(rename = "smsp__inst_issued_sum")]
    pub smsp_inst_issued_sum: Metric<Float>,
    #[serde(rename = "smsp__issue_active_avg_per_active_cycle")]
    pub smsp_issue_active_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "smsp__issue_active_per_active_cycle_pct")]
    pub smsp_issue_active_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "smsp__maximum_warps_avg_per_active_cycle")]
    pub smsp_maximum_warps_avg_per_active_cycle: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_aggregated_passes")]
    pub smsp_pcsamp_aggregated_passes: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_buffer_size_bytes")]
    pub smsp_pcsamp_buffer_size_bytes: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_dropped_bytes")]
    pub smsp_pcsamp_dropped_bytes: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_interval")]
    pub smsp_pcsamp_interval: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_interval_cycles")]
    pub smsp_pcsamp_interval_cycles: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_allocation_stall")]
    pub smsp_pcsamp_warp_stall_allocation_stall: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_barrier")]
    pub smsp_pcsamp_warp_stall_barrier: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_dispatch_stall")]
    pub smsp_pcsamp_warp_stall_dispatch_stall: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_drain")]
    pub smsp_pcsamp_warp_stall_drain: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_imc_miss")]
    pub smsp_pcsamp_warp_stall_imc_miss: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_long_scoreboard")]
    pub smsp_pcsamp_warp_stall_long_scoreboard: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_math_pipe_throttle")]
    pub smsp_pcsamp_warp_stall_math_pipe_throttle: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_membar")]
    pub smsp_pcsamp_warp_stall_membar: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_mio_throttle")]
    pub smsp_pcsamp_warp_stall_mio_throttle: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_misc")]
    pub smsp_pcsamp_warp_stall_misc: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_no_instructions")]
    pub smsp_pcsamp_warp_stall_no_instructions: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_not_selected")]
    pub smsp_pcsamp_warp_stall_not_selected: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_selected")]
    pub smsp_pcsamp_warp_stall_selected: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_short_scoreboard")]
    pub smsp_pcsamp_warp_stall_short_scoreboard: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_tex_throttle")]
    pub smsp_pcsamp_warp_stall_tex_throttle: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_tile_allocation_stall")]
    pub smsp_pcsamp_warp_stall_tile_allocation_stall: Metric<Float>,
    #[serde(rename = "smsp__pcsamp_warp_stall_wait")]
    pub smsp_pcsamp_warp_stall_wait: Metric<Float>,
    #[serde(rename = "smsp__thread_inst_executed_not_pred_off_per_inst_executed")]
    pub smsp_thread_inst_executed_not_pred_off_per_inst_executed: Metric<Float>,
    #[serde(rename = "smsp__thread_inst_executed_per_inst_executed")]
    pub smsp_thread_inst_executed_per_inst_executed: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_inst_executed")]
    pub smsp_warp_cycles_per_inst_executed: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_inst_issued")]
    pub smsp_warp_cycles_per_inst_issued: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_active")]
    pub smsp_warp_cycles_per_issue_active: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_allocation_stall")]
    pub smsp_warp_cycles_per_issue_stall_allocation_stall: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_barrier")]
    pub smsp_warp_cycles_per_issue_stall_barrier: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_dispatch_stall")]
    pub smsp_warp_cycles_per_issue_stall_dispatch_stall: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_drain")]
    pub smsp_warp_cycles_per_issue_stall_drain: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_imc_miss")]
    pub smsp_warp_cycles_per_issue_stall_imc_miss: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_long_scoreboard")]
    pub smsp_warp_cycles_per_issue_stall_long_scoreboard: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_math_pipe_throttle")]
    pub smsp_warp_cycles_per_issue_stall_math_pipe_throttle: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_membar")]
    pub smsp_warp_cycles_per_issue_stall_membar: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_mio_throttle")]
    pub smsp_warp_cycles_per_issue_stall_mio_throttle: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_misc")]
    pub smsp_warp_cycles_per_issue_stall_misc: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_no_instructions")]
    pub smsp_warp_cycles_per_issue_stall_no_instructions: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_not_selected")]
    pub smsp_warp_cycles_per_issue_stall_not_selected: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_selected")]
    pub smsp_warp_cycles_per_issue_stall_selected: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_short_scoreboard")]
    pub smsp_warp_cycles_per_issue_stall_short_scoreboard: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_tex_throttle")]
    pub smsp_warp_cycles_per_issue_stall_tex_throttle: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_tile_allocation_stall")]
    pub smsp_warp_cycles_per_issue_stall_tile_allocation_stall: Metric<Float>,
    #[serde(rename = "smsp__warp_cycles_per_issue_stall_wait")]
    pub smsp_warp_cycles_per_issue_stall_wait: Metric<Float>,
    #[serde(rename = "smsp__warps_per_cycle_max")]
    pub smsp_warps_per_cycle_max: Metric<Float>,
    #[serde(rename = "tex__global_ld_unique_sector_requests")]
    pub tex_global_ld_unique_sector_requests: Metric<Float>,
    #[serde(rename = "tex__hitrate_pct")]
    pub tex_hitrate_pct: Metric<Float>,
    #[serde(rename = "tex__local_ld_unique_sector_requests")]
    pub tex_local_ld_unique_sector_requests: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_global_atom")]
    pub tex_m_rd_bytes_global_atom: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_global_atom_per_sec")]
    pub tex_m_rd_bytes_global_atom_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_miss_global_ld_cached")]
    pub tex_m_rd_bytes_miss_global_ld_cached: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_miss_global_ld_uncached")]
    pub tex_m_rd_bytes_miss_global_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_miss_local_ld_cached")]
    pub tex_m_rd_bytes_miss_local_ld_cached: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_miss_local_ld_uncached")]
    pub tex_m_rd_bytes_miss_local_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_miss_surface_ld")]
    pub tex_m_rd_bytes_miss_surface_ld: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_surface_atom")]
    pub tex_m_rd_bytes_surface_atom: Metric<Float>,
    #[serde(rename = "tex__m_rd_bytes_surface_atom_per_sec")]
    pub tex_m_rd_bytes_surface_atom_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_global_atom")]
    pub tex_m_rd_sectors_global_atom: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_global_atom_pct")]
    pub tex_m_rd_sectors_global_atom_pct: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_cached")]
    pub tex_m_rd_sectors_miss_global_ld_cached: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_cached_pct")]
    pub tex_m_rd_sectors_miss_global_ld_cached_pct: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_uncached")]
    pub tex_m_rd_sectors_miss_global_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_global_ld_uncached_pct")]
    pub tex_m_rd_sectors_miss_global_ld_uncached_pct: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_cached")]
    pub tex_m_rd_sectors_miss_local_ld_cached: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_cached_pct")]
    pub tex_m_rd_sectors_miss_local_ld_cached_pct: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_uncached")]
    pub tex_m_rd_sectors_miss_local_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_local_ld_uncached_pct")]
    pub tex_m_rd_sectors_miss_local_ld_uncached_pct: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_surface_ld")]
    pub tex_m_rd_sectors_miss_surface_ld: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_miss_surface_ld_pct")]
    pub tex_m_rd_sectors_miss_surface_ld_pct: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_surface_atom")]
    pub tex_m_rd_sectors_surface_atom: Metric<Float>,
    #[serde(rename = "tex__m_rd_sectors_surface_atom_pct")]
    pub tex_m_rd_sectors_surface_atom_pct: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_global_atom")]
    pub tex_m_wr_bytes_global_atom: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_global_atom_per_sec")]
    pub tex_m_wr_bytes_global_atom_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_global_nonatom")]
    pub tex_m_wr_bytes_global_nonatom: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_global_nonatom_per_sec")]
    pub tex_m_wr_bytes_global_nonatom_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_global_red")]
    pub tex_m_wr_bytes_global_red: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_global_red_per_sec")]
    pub tex_m_wr_bytes_global_red_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_local_st")]
    pub tex_m_wr_bytes_local_st: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_local_st_per_sec")]
    pub tex_m_wr_bytes_local_st_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_surface_atom")]
    pub tex_m_wr_bytes_surface_atom: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_surface_atom_per_sec")]
    pub tex_m_wr_bytes_surface_atom_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_surface_nonatom")]
    pub tex_m_wr_bytes_surface_nonatom: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_surface_nonatom_per_sec")]
    pub tex_m_wr_bytes_surface_nonatom_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_surface_red")]
    pub tex_m_wr_bytes_surface_red: Metric<Float>,
    #[serde(rename = "tex__m_wr_bytes_surface_red_per_sec")]
    pub tex_m_wr_bytes_surface_red_per_sec: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_global_atom")]
    pub tex_m_wr_sectors_global_atom: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_global_atom_pct")]
    pub tex_m_wr_sectors_global_atom_pct: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_global_nonatom")]
    pub tex_m_wr_sectors_global_nonatom: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_global_nonatom_pct")]
    pub tex_m_wr_sectors_global_nonatom_pct: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_global_red")]
    pub tex_m_wr_sectors_global_red: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_global_red_pct")]
    pub tex_m_wr_sectors_global_red_pc: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_local_st")]
    pub tex_m_wr_sectors_local_st: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_local_st_pct")]
    pub tex_m_wr_sectors_local_st_pct: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_surface_atom")]
    pub tex_m_wr_sectors_surface_atom: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_surface_atom_pct")]
    pub tex_m_wr_sectors_surface_atom_pct: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_surface_nonatom")]
    pub tex_m_wr_sectors_surface_nonatom: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_surface_nonatom_pct")]
    pub tex_m_wr_sectors_surface_nonatom_pct: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_surface_red")]
    pub tex_m_wr_sectors_surface_red: Metric<Float>,
    #[serde(rename = "tex__m_wr_sectors_surface_red_pct")]
    pub tex_m_wr_sectors_surface_red_pct: Metric<Float>,
    #[serde(rename = "tex__sol_pct")]
    pub tex_sol_pct: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_cached")]
    pub tex_t_bytes_miss_global_ld_cached: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_cached_per_sec")]
    pub tex_t_bytes_miss_global_ld_cached_per_sec: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_uncached")]
    pub tex_t_bytes_miss_global_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_global_ld_uncached_per_sec")]
    pub tex_t_bytes_miss_global_ld_uncached_per_sec: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_cached")]
    pub tex_t_bytes_miss_local_ld_cached: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_cached_per_sec")]
    pub tex_t_bytes_miss_local_ld_cached_per_sec: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_uncached")]
    pub tex_t_bytes_miss_local_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_local_ld_uncached_per_sec")]
    pub tex_t_bytes_miss_local_ld_uncached_per_sec: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_surface_ld")]
    pub tex_t_bytes_miss_surface_ld: Metric<Float>,
    #[serde(rename = "tex__t_bytes_miss_surface_ld_per_sec")]
    pub tex_t_bytes_miss_surface_ld_per_sec: Metric<Float>,
    #[serde(rename = "tex__t_sectors_miss_global_ld_cached")]
    pub tex_t_sectors_miss_global_ld_cached: Metric<Float>,
    #[serde(rename = "tex__t_sectors_miss_global_ld_uncached")]
    pub tex_t_sectors_miss_global_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__t_sectors_miss_local_ld_cached")]
    pub tex_t_sectors_miss_local_ld_cached: Metric<Float>,
    #[serde(rename = "tex__t_sectors_miss_local_ld_uncached")]
    pub tex_t_sectors_miss_local_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__t_sectors_miss_surface_ld")]
    pub tex_t_sectors_miss_surface_ld: Metric<Float>,
    #[serde(rename = "tex__tex2sm_tex_nonatomic_active")]
    pub tex_tex2sm_tex_nonatomic_active: Metric<Float>,
    #[serde(rename = "tex__tex2sm_tex_nonatomic_utilization")]
    pub tex_tex2sm_tex_nonatomic_utilization: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_atom")]
    pub tex_texin_requests_global_atom: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_atom_per_active_cycle_pct")]
    pub tex_texin_requests_global_atom_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_atomcas")]
    pub tex_texin_requests_global_atomcas: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_atomcas_per_active_cycle_pct")]
    pub tex_texin_requests_global_atomcas_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_ld_cached")]
    pub tex_texin_requests_global_ld_cached: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_ld_cached_per_active_cycle_pct")]
    pub tex_texin_requests_global_ld_cached_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_ld_uncached")]
    pub tex_texin_requests_global_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_ld_uncached_per_active_cycle_pct")]
    pub tex_texin_requests_global_ld_uncached_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_red")]
    pub tex_texin_requests_global_red: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_red_per_active_cycle_pct")]
    pub tex_texin_requests_global_red_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_st")]
    pub tex_texin_requests_global_st: Metric<Float>,
    #[serde(rename = "tex__texin_requests_global_st_per_active_cycle_pct")]
    pub tex_texin_requests_global_st_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_local_ld_cached")]
    pub tex_texin_requests_local_ld_cached: Metric<Float>,
    #[serde(rename = "tex__texin_requests_local_ld_cached_per_active_cycle_pct")]
    pub tex_texin_requests_local_ld_cached_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_local_ld_uncached")]
    pub tex_texin_requests_local_ld_uncached: Metric<Float>,
    #[serde(rename = "tex__texin_requests_local_ld_uncached_per_active_cycle_pct")]
    pub tex_texin_requests_local_ld_uncached_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_local_st")]
    pub tex_texin_requests_local_st: Metric<Float>,
    #[serde(rename = "tex__texin_requests_local_st_per_active_cycle_pct")]
    pub tex_texin_requests_local_st_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_atom")]
    pub tex_texin_requests_surface_atom: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_atom_per_active_cycle_pct")]
    pub tex_texin_requests_surface_atom_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_atomcas")]
    pub tex_texin_requests_surface_atomcas: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_atomcas_per_active_cycle_pct")]
    pub tex_texin_requests_surface_atomcas_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_ld")]
    pub tex_texin_requests_surface_ld: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_ld_per_active_cycle_pct")]
    pub tex_texin_requests_surface_ld_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_red")]
    pub tex_texin_requests_surface_red: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_red_per_active_cycle_pct")]
    pub tex_texin_requests_surface_red_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_st")]
    pub tex_texin_requests_surface_st: Metric<Float>,
    #[serde(rename = "tex__texin_requests_surface_st_per_active_cycle_pct")]
    pub tex_texin_requests_surface_st_per_active_cycle_pct: Metric<Float>,
    #[serde(rename = "thread_inst_executed_true")]
    pub thread_inst_executed_true: Metric<String>,
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
