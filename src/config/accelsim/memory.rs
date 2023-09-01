use super::Boolean;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[clap()]
pub struct AddressMapping {
    #[clap(
        long = "gpgpu_mem_addr_mapping",
        help = "mapping memory address to dram model {dramid@<start bit>;<memory address map>}"
    )]
    pub addrdec_option: Option<String>,
    #[clap(
        long = "gpgpu_mem_addr_test",
        help = "run sweep test to check address mapping for aliased address",
        default_value = "0"
    )]
    pub run_test: Boolean,
    #[clap(
        long = "gpgpu_mem_address_mask",
        help = "0 = old addressing mask, 1 = new addressing mask, 2 = new add. mask + flipped bank sel and chip sel bits",
        default_value = "0"
    )]
    pub gpgpu_mem_address_mask: u32,
    #[clap(
        long = "gpgpu_memory_partition_indexing",
        help = "0 = consecutive (no indexing), 1 = bitwise xoring, 2 = IPoly, 3 = pae, 4 = random, 5 = custom",
        default_value = "0"
    )]
    pub memory_partition_indexing: u32,
}

impl Default for AddressMapping {
    fn default() -> Self {
        Self {
            addrdec_option: None,
            run_test: false.into(),
            gpgpu_mem_address_mask: 0,
            memory_partition_indexing: 0,
        }
    }
}

#[derive(Parser, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[clap()]
pub struct MemoryConfig {
    #[clap(
        long = "gpgpu_perf_sim_memcpy",
        help = "Fill the L2 cache on memcpy",
        default_value = "1"
    )]
    pub perf_sim_memcpy: Boolean,
    #[clap(
        long = "gpgpu_simple_dram_model",
        help = "simple_dram_model with fixed latency and BW",
        default_value = "0"
    )]
    pub simple_dram_model: Boolean,
    #[clap(
        long = "gpgpu_dram_scheduler",
        help = "0 = fifo, 1 = FR-FCFS (defaul)",
        default_value = "1"
    )]
    pub scheduler_type: u32,
    #[clap(
        long = "gpgpu_dram_partition_queues",
        help = "i2$:$2d:d2$:$2i",
        default_value = "8:8:8:8"
    )]
    pub gpgpu_l2_queue_config: String,
    #[clap(
        long = "l2_ideal",
        help = "Use a ideal L2 cache that always hit",
        default_value = "0"
    )]
    pub l2_ideal: Boolean,
    #[clap(
        long = "gpgpu_cache:dl2",
        help = "unified banked L2 data cache config {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<merge>,<mq>}",
        default_value = "64:128:8,L:B:m:N,A:16:4,4"
    )]
    pub l2_config_string: String,
    #[clap(
        long = "gpgpu_cache:dl2_texture_only",
        help = "L2 cache used for texture only",
        default_value = "1"
    )]
    pub l2_texure_only: Boolean,
    #[clap(
        long = "gpgpu_n_mem",
        help = "number of memory modules (e.g. memory controllers) in gpu",
        default_value = "8"
    )]
    pub n_mem: u32,
    #[clap(
        long = "gpgpu_n_sub_partition_per_mchannel",
        help = "number of memory subpartition in each memory module",
        default_value = "1"
    )]
    pub n_sub_partition_per_memory_channel: u32,
    #[clap(
        long = "gpgpu_n_mem_per_ctrlr",
        help = "number of memory chips per memory controller",
        default_value = "1"
    )]
    pub gpu_n_mem_per_ctrlr: u32,
    #[clap(
        long = "gpgpu_memlatency_stat",
        help = "track and display latency statistics 0x2 enables MC, 0x4 enables queue logs",
        default_value = "0"
    )]
    pub gpgpu_memlatency_stat: u32,
    #[clap(
        long = "gpgpu_frfcfs_dram_sched_queue_size",
        help = "0 = unlimited (default); # entries per chip",
        default_value = "0"
    )]
    pub gpgpu_frfcfs_dram_sched_queue_size: u32,
    #[clap(
        long = "gpgpu_dram_return_queue_size",
        help = "0 = unlimited (default); # entries per chip",
        default_value = "0"
    )]
    pub gpgpu_dram_return_queue_size: u32,
    #[clap(
        long = "gpgpu_dram_buswidth",
        help = "default = 4 bytes (8 bytes per cycle at DDR)",
        default_value = "4"
    )]
    pub dram_bus_width: u32,
    #[clap(
        long = "gpgpu_dram_burst_length",
        help = "Burst length of each DRAM request (default = 4 data bus cycle)",
        default_value = "4"
    )]
    pub dram_burst_length: u32,
    #[clap(
        long = "dram_data_command_freq_ratio",
        help = "Frequency ratio between DRAM data bus and command bus (default = 2 times, i.e. DDR)",
        default_value = "2"
    )]
    pub data_command_freq_ratio: u32,
    #[clap(
        long = "gpgpu_dram_timing_opt",
        help = "DRAM timing parameters = {nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
        default_value = "4:2:8:12:21:13:34:9:4:5:13:1:0:0"
    )]
    pub gpgpu_dram_timing_opt: String,
    #[clap(
        long = "gpgpu_l2_rop_latency",
        help = "ROP queue latency (default 85)",
        default_value = "85"
    )]
    pub rop_latency: u32,
    #[clap(
        long = "dram_latency",
        help = "DRAM latency (default 30)",
        default_value = "30"
    )]
    pub dram_latency: u32,
    #[clap(
        long = "dram_dual_bus_interface",
        help = "dual_bus_interface (default = 0)",
        default_value = "0"
    )]
    pub dual_bus_interface: u32,
    #[clap(
        long = "dram_bnk_indexing_policy",
        help = "dram_bnk_indexing_policy (0 = normal indexing, 1 = Xoring with the higher bits) (Default = 0)",
        default_value = "0"
    )]
    pub dram_bnk_indexing_policy: u32,
    #[clap(
        long = "dram_bnkgrp_indexing_policy",
        help = "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 = take lower bits) (Default = 0)",
        default_value = "0"
    )]
    pub dram_bnkgrp_indexing_policy: u32,
    #[clap(
        long = "dram_seperate_write_queue_enable",
        help = "Seperate_Write_Queue_Enable",
        default_value = "0"
    )]
    pub seperate_write_queue_enabled: Boolean,
    #[clap(
        long = "dram_write_queue_size",
        help = "Write_Queue_Size",
        default_value = "32:28:16"
    )]
    pub write_queue_size_opt: String,
    #[clap(
        long = "dram_elimnate_rw_turnaround",
        help = "elimnate_rw_turnaround i.e set tWTR and tRTW = 0",
        default_value = "0"
    )]
    pub elimnate_rw_turnaround: Boolean,
    #[clap(long = "icnt_flit_size", help = "icnt_flit_size", default_value = "32")]
    pub icnt_flit_size: u32,

    #[clap(flatten)]
    pub address_mapping: AddressMapping,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            perf_sim_memcpy: true.into(),
            simple_dram_model: false.into(),
            scheduler_type: 1,
            gpgpu_l2_queue_config: "8:8:8:8".to_string(),
            l2_ideal: false.into(),
            l2_config_string: "64:128:8,L:B:m:N,A:16:4,4".to_string(),
            l2_texure_only: true.into(),
            n_mem: 8,
            n_sub_partition_per_memory_channel: 1,
            gpu_n_mem_per_ctrlr: 1,
            gpgpu_memlatency_stat: 0,
            gpgpu_frfcfs_dram_sched_queue_size: 0,
            gpgpu_dram_return_queue_size: 0,
            dram_bus_width: 4,
            dram_burst_length: 4,
            data_command_freq_ratio: 2,
            gpgpu_dram_timing_opt: "4:2:8:12:21:13:34:9:4:5:13:1:0:0".to_string(),
            rop_latency: 85,
            dram_latency: 30,
            dual_bus_interface: 0,
            dram_bnk_indexing_policy: 0,
            dram_bnkgrp_indexing_policy: 0,
            seperate_write_queue_enabled: false.into(),
            write_queue_size_opt: "32:28:16".to_string(),
            elimnate_rw_turnaround: false.into(),
            icnt_flit_size: 32,
            address_mapping: AddressMapping::default(),
        }
    }
}
