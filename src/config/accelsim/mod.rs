#![allow(clippy::module_name_repetitions)]

pub mod core;
pub mod dram;
pub mod functional;
pub mod interconnect;
pub mod memory;
pub mod ptx;
pub mod sim;
pub mod trace;

use clap::Parser;
use color_eyre::eyre;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct Boolean(bool);

impl From<Boolean> for bool {
    fn from(b: Boolean) -> Self {
        b.0
    }
}

impl From<bool> for Boolean {
    fn from(b: bool) -> Self {
        Self(b)
    }
}

impl std::str::FromStr for Boolean {
    type Err = std::num::ParseIntError;

    fn from_str(arg: &str) -> Result<Self, Self::Err> {
        match arg.to_lowercase().trim() {
            "t" | "true" | "yes" => Ok(Boolean(true)),
            "f" | "false" | "no" => Ok(Boolean(false)),
            _ => {
                let arg: i32 = arg.trim().parse()?;
                Ok(Boolean(arg != 0))
            }
        }
    }
}

#[derive(Parser, Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[clap(
    trailing_var_arg = true,
    // allow_hyphen_values = true,
    // arg_required_else_help = false
)]
pub struct Config {
    // #[clap(short, help = "todo")]
    // pub gpgpu_ptx_instruction_classification: Option<usize>,
    #[clap(flatten)]
    pub shader_core: core::CoreConfig,

    #[clap(flatten)]
    pub ptx: ptx::PTXConfig,

    #[clap(flatten)]
    pub trace: trace::TraceConfig,

    #[clap(flatten)]
    pub sim: sim::SimConfig,

    #[clap(flatten)]
    pub dram_timing: dram::TimingConfig,

    #[clap(flatten)]
    pub functional: functional::FunctionalConfig,

    #[clap(flatten)]
    pub interconn: interconnect::InterconnectConfig,

    #[clap(flatten)]
    pub memory: memory::MemoryConfig,

    #[clap(num_args(0..), allow_hyphen_values = true)]
    pub unknown: Vec<String>,
}

use once_cell::sync::Lazy;
use regex::Regex;

static ARGUMENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    let arg = r"([\w\d\-:]+)";
    let single_quoted_string = "(?:'(?:[^\\']|\\.)*')";
    let double_quoted_string = r#"(?:"(?:[^\"]|\\.)*")"#;
    let value_excluding_comment = r"(?:[^#\n]+)";
    let trailing_comment = r"(?:#.*)?";
    let pattern = [
        r"^\s*-{1,2}",
        arg,
        r"\s+(",
        single_quoted_string,
        "|",
        double_quoted_string,
        "|",
        value_excluding_comment,
        ")",
        trailing_comment,
    ];
    let pattern = pattern.join("");
    regex::RegexBuilder::new(&pattern)
        .multi_line(true)
        .build()
        .unwrap()
});

pub fn extract_arguments(config: &str) -> impl Iterator<Item = (&str, &str)> {
    ARGUMENT_REGEX.captures_iter(config).filter_map(|cap| {
        let key = cap.get(1)?.as_str().trim();
        let value = cap.get(2)?.as_str().trim();
        Some((key, value))
    })
}

impl Config {
    pub fn parse(config: impl AsRef<str>) -> eyre::Result<Self> {
        let args =
            extract_arguments(config.as_ref()).map(|(key, value)| format!("--{key}={value}"));
        let args: Vec<String> = ["test".to_string()].into_iter().chain(args).collect();
        let config = Self::try_parse_from(args)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use clap::Parser;
    use color_eyre::eyre;
    use std::path::PathBuf;

    #[test]
    fn test_read_trace_config_file_gtx1080() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.trace.config");
        let config = std::fs::read_to_string(config_path)?;
        diff::assert_eq!(have: super::Config::parse("")?, want: super::Config::default());
        diff::assert_eq!(
            have: super::Config::parse(config)?,
            want: super::Config {
                shader_core: super::core::CoreConfig {
                    specialized_unit_1: "0,4,4,4,4,BRA".to_string(),
                    specialized_unit_2: "0,4,4,4,4,BRA".to_string(),
                    specialized_unit_3: "0,4,4,4,4,BRA".to_string(),
                    specialized_unit_4: "0,4,4,4,4,BRA".to_string(),
                    ..super::core::CoreConfig::default()
                },
                trace: super::trace::TraceConfig {
                    trace_opcode_latency_initiation_int: "6,1".to_string(),
                    trace_opcode_latency_initiation_sp: "6,1".to_string(),
                    trace_opcode_latency_initiation_dp: "8,8".to_string(),
                    trace_opcode_latency_initiation_sfu: "14,4".to_string(),
                    trace_opcode_latency_initiation_tensor: "4,1".to_string(),
                    trace_opcode_latency_initiation_spec_op_1: "4,4".to_string(),
                    trace_opcode_latency_initiation_spec_op_2: "4,4".to_string(),
                    trace_opcode_latency_initiation_spec_op_3: "4,4".to_string(),
                    trace_opcode_latency_initiation_spec_op_4: "4,4".to_string(),
                    ..super::trace::TraceConfig::default()
                },
                ..super::Config::default()
            },
        );
        Ok(())
    }

    #[test]
    fn test_read_config_file_gtx1080() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.original.config");
        let config = std::fs::read_to_string(config_path)?;
        diff::assert_eq!(have: super::Config::parse("")?, want: super::Config::default());

        diff::assert_eq!(
            have: super::Config::try_parse_from(vec![] as Vec<String>)?,
            want: super::Config::default()
        );

        let mut have = super::Config::parse(config)?;
        dbg!(&have);
        let mut want = super::Config {
            functional: super::functional::FunctionalConfig {
                m_ptx_force_max_capability: 61,
                m_ptx_convert_to_ptxplus: false.into(),
                ..super::functional::FunctionalConfig::default()
            },
            sim: super::sim::SimConfig {
                gpgpu_ptx_instruction_classification: 0,
                g_ptx_sim_mode: 0,
                gpgpu_runtime_stat: "500".to_string(),
                gpgpu_clock_domains: "1417.0:1417.0:1417.0:2500.0".to_string(),
                gpgpu_compute_capability_major: 6,
                gpgpu_compute_capability_minor: 1,
                gpgpu_flush_l1_cache: true.into(),
                g_kernel_launch_latency: 5000,
                ..super::sim::SimConfig::default()
            },
            memory: super::memory::MemoryConfig {
                n_mem: 12,
                n_sub_partition_per_memory_channel: 2,
                gpu_n_mem_per_ctrlr: 1,
                gpgpu_l2_queue_config: "32:32:32:32".into(),
                l2_config_string: "S:64:128:16,L:B:m:W:P,A:256:64,16:0,32".into(),
                l2_texure_only: false.into(),
                rop_latency: 120,
                dram_latency: 100,
                scheduler_type: 1,
                // TODO: dedup?
                dram_bnk_indexing_policy: 0,
                dram_bnkgrp_indexing_policy: 1,
                gpgpu_frfcfs_dram_sched_queue_size: 64,
                gpgpu_dram_return_queue_size: 64,
                dram_bus_width: 4,
                dram_burst_length: 8,
                gpgpu_memlatency_stat: 14,
                data_command_freq_ratio: 4,
                gpgpu_dram_timing_opt: r#""nbk=16:CCD=2:RRD=8:RCD=16:RAS=37:RP=16:RC=52:
                        CL=16:WL=6:CDLR=7:WR=16:nbkgrp=4:CCDL=4:RTPL=3""#.to_string(),
                address_mapping: super::memory::AddressMapping {
                    gpgpu_mem_address_mask: 1,
                    addrdec_option: Some("dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS".to_string()),
                    memory_partition_indexing: 4,
                    ..super::memory::AddressMapping::default()
                },
                ..super::memory::MemoryConfig::default()
            },
            ptx: super::ptx::PTXConfig {
                opcode_latency_int: "4,13,4,5,145,32".to_string(),
                opcode_initiation_int: "1,1,1,1,4,4".to_string(),
                opcode_latency_fp: "4,13,4,4,39".to_string(),
                opcode_initiation_fp: "1,2,1,1,4".to_string(),
                opcode_latency_dp: "8,19,8,8,330".to_string(),
                opcode_initiation_dp: "8,8,8,8,130".to_string(),
                opcode_latency_sfu: "20".to_string(),
                opcode_initiation_sfu: "4".to_string(),
                g_occupancy_sm_number: 62,
                g_ptx_save_converted_ptxplus: false.into(),
                ..super::ptx::PTXConfig::default()
            },
            shader_core: super::core::CoreConfig {
                gpgpu_n_clusters: 28,
                gpgpu_n_cores_per_cluster: 1,
                gpgpu_n_cluster_ejection_buffer_size: 32,
                gpgpu_ignore_resources_limitation: true.into(),

                gpgpu_shader_registers: 65536,
                gpgpu_shader_core_pipeline: "2048:32".to_string(),
                gpgpu_shader_cta: 32,
                gpgpu_sub_core_model: true.into(),
                gpgpu_simd_model: 1,
                gpgpu_pipeline_widths: "4,0,0,4,4,4,0,0,4,4,8".to_string(),
                gpgpu_num_sp_units: 4,
                gpgpu_num_sfu_units: 4,
                gpgpu_tensor_core_avail: 0,
                gpgpu_num_tensor_core_units: 0,

                gpgpu_enable_specialized_operand_collector: false.into(),
                gpgpu_operand_collector_num_units_gen: 8,
                gpgpu_operand_collector_num_in_ports_gen: 8,
                gpgpu_operand_collector_num_out_ports_gen: 8,

                gpgpu_operand_collector_num_units_sp: 4,
                gpgpu_operand_collector_num_units_sfu: 4,
                gpgpu_operand_collector_num_units_mem: 2,
                gpgpu_operand_collector_num_in_ports_sp: 1,
                gpgpu_operand_collector_num_out_ports_sp: 1,
                gpgpu_operand_collector_num_in_ports_sfu: 1,
                gpgpu_operand_collector_num_out_ports_sfu: 1,
                gpgpu_operand_collector_num_in_ports_mem: 1,
                gpgpu_operand_collector_num_out_ports_mem: 1,
                gpgpu_coalesce_arch: 61,
                gpgpu_num_reg_banks: 16,
                gpgpu_reg_file_port_throughput: 2,
                gpgpu_shmem_num_banks: 32,
                gpgpu_shmem_limited_broadcast: 0,
                gpgpu_shmem_warp_parts: 1,
                gpgpu_max_insn_issue_per_warp: 2,
                gpgpu_num_sched_per_core: 4,
                gpgpu_scheduler: "gto".to_string(),

                gpgpu_cache_dl1: "S:4:128:96,L:L:s:N:L,A:256:8,16:0,32".to_string(),
                gpgpu_cache_dl1_pref_l1: "S:4:128:96,L:L:s:N:L,A:256:8,16:0,32".to_string(),
                gpgpu_cache_dl1_pref_shared: "S:4:128:96,L:L:s:N:L,A:256:8,16:0,32".to_string(),
                gpgpu_cache_il1: "N:8:128:4,L:R:f:N:L,S:2:48,4".to_string(),
                gpgpu_tex_cache_l1: "N:16:128:24,L:R:m:N:L,T:128:4,128:2".to_string(),
                gpgpu_const_cache_l1: "N:128:64:2,L:R:f:N:L,S:2:64,4".to_string(),
                gpgpu_perfect_inst_const_cache: true.into(),
                gpgpu_inst_fetch_throughput: 8,
                gpgpu_l1_banks: 2,
                gpgpu_l1_latency: 82,
                gpgpu_smem_latency: 24,
                gpgpu_gmem_skip_l1d: false.into(),
                gpgpu_clock_gated_lanes: true.into(),
                gpgpu_shmem_size: 98304,
                gpgpu_shmem_size_default: 98304,
                gpgpu_shmem_size_pref_l1: 98304,
                gpgpu_shmem_size_pref_shared: 98304,
                ..super::core::CoreConfig::default()
            },
            interconn: super::interconnect::InterconnectConfig {
                g_network_mode: 1,
                g_network_config_filename: "config_pascal_islip.icnt".to_string(),
                icnt_flit_size: 40,
                ..super::interconnect::InterconnectConfig::default()
            },
            ..super::Config::default()
        };
        have.unknown.clear();
        want.unknown.clear();
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }

    #[test]
    fn test_extract_arguments() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.original.config");
        let config = std::fs::read_to_string(config_path)?;
        let have: Vec<_> = super::extract_arguments(&config).collect();
        dbg!(&have);
        let want = vec![
            (
                "gpgpu_ptx_instruction_classification",
                "0",
            ),
            (
                "gpgpu_ptx_sim_mode",
                "0",
            ),
            (
                "gpgpu_ptx_force_max_capability",
                "61",
            ),
            (
                "gpgpu_ignore_resources_limitation",
                "1",
            ),
            (
                "gpgpu_stack_size_limit",
                "1024",
            ),
            (
                "gpgpu_heap_size_limit",
                "8388608",
            ),
            (
                "gpgpu_runtime_sync_depth_limit",
                "2",
            ),
            (
                "gpgpu_runtime_pending_launch_count_limit",
                "2048",
            ),
            (
                "gpgpu_kernel_launch_latency",
                "5000",
            ),
            (
                "gpgpu_compute_capability_major",
                "6",
            ),
            (
                "gpgpu_compute_capability_minor",
                "1",
            ),
            (
                "gpgpu_ptx_convert_to_ptxplus",
                "0",
            ),
                    (
                "gpgpu_ptx_save_converted_ptxplus",
                "0",
            ),
            (
                "gpgpu_n_clusters",
                "28",
            ),
            (
                "gpgpu_n_cores_per_cluster",
                "1",
            ),
            (
                "gpgpu_n_mem",
                "12",
            ),
            (
                "gpgpu_n_sub_partition_per_mchannel",
                "2",
            ),
            (
                "gpgpu_clock_gated_lanes",
                "1",
            ),
            (
                "gpgpu_clock_domains",
                "1417.0:1417.0:1417.0:2500.0",
            ),
            (
                "gpgpu_shader_registers",
                "65536",
            ),
            (
                "gpgpu_occupancy_sm_number",
                "62",
            ),
            (
                "gpgpu_shader_core_pipeline",
                "2048:32",
            ),
            (
                "gpgpu_shader_cta",
                "32",
            ),
            (
                "gpgpu_simd_model",
                "1",
            ),
                    (
                "gpgpu_pipeline_widths",
                "4,0,0,4,4,4,0,0,4,4,8",
            ),
            (
                "gpgpu_num_sp_units",
                "4",
            ),
            (
                "gpgpu_num_sfu_units",
                "4",
            ),
            (
                "ptx_opcode_latency_int",
                "4,13,4,5,145,32",
            ),
            (
                "ptx_opcode_initiation_int",
                "1,1,1,1,4,4",
            ),
            (
                "ptx_opcode_latency_fp",
                "4,13,4,4,39",
            ),
            (
                "ptx_opcode_initiation_fp",
                "1,2,1,1,4",
            ),
            (
                "ptx_opcode_latency_dp",
                "8,19,8,8,330",
            ),
            (
                "ptx_opcode_initiation_dp",
                "8,8,8,8,130",
            ),
            (
                "ptx_opcode_initiation_sfu",
                "4",
            ),
            (
                "ptx_opcode_latency_sfu",
                "20",
            ),
            (
                "gpgpu_sub_core_model",
                "1",
            ),
                    (
                "gpgpu_enable_specialized_operand_collector",
                "0",
            ),
            (
                "gpgpu_operand_collector_num_units_gen",
                "8",
            ),
            (
                "gpgpu_operand_collector_num_in_ports_gen",
                "8",
            ),
            (
                "gpgpu_operand_collector_num_out_ports_gen",
                "8",
            ),
            (
                "gpgpu_num_reg_banks",
                "16",
            ),
            (
                "gpgpu_reg_file_port_throughput",
                "2",
            ),
            (
                "gpgpu_shmem_num_banks",
                "32",
            ),
            (
                "gpgpu_shmem_limited_broadcast",
                "0",
            ),
            (
                "gpgpu_shmem_warp_parts",
                "1",
            ),
            (
                "gpgpu_coalesce_arch",
                "61",
            ),
            (
                "gpgpu_num_sched_per_core",
                "4",
            ),
            (
                "gpgpu_scheduler",
                "gto",
            ),
                    (
                "gpgpu_max_insn_issue_per_warp",
                "2",
            ),
            (
                "gpgpu_dual_issue_diff_exec_units",
                "1",
            ),
            (
                "gpgpu_l1_banks",
                "2",
            ),
            (
                "gpgpu_cache:dl1",
                "S:4:128:96,L:L:s:N:L,A:256:8,16:0,32",
            ),
            (
                "gpgpu_cache:dl1PrefL1",
                "S:4:128:96,L:L:s:N:L,A:256:8,16:0,32",
            ),
            (
                "gpgpu_cache:dl1PrefShared",
                "S:4:128:96,L:L:s:N:L,A:256:8,16:0,32",
            ),
            (
                "gpgpu_shmem_size",
                "98304",
            ),
            (
                "gpgpu_shmem_sizeDefault",
                "98304",
            ),
            (
                "gpgpu_shmem_size_PrefL1",
                "98304",
            ),
            (
                "gpgpu_shmem_size_PrefShared",
                "98304",
            ),
            (
                "gpgpu_gmem_skip_L1D",
                "0",
            ),
            (
                "icnt_flit_size",
                "40",
            ),
                    (
                "gpgpu_n_cluster_ejection_buffer_size",
                "32",
            ),
            (
                "gpgpu_l1_latency",
                "82",
            ),
            (
                "gpgpu_smem_latency",
                "24",
            ),
            (
                "gpgpu_flush_l1_cache",
                "1",
            ),
            (
                "gpgpu_cache:dl2",
                "S:64:128:16,L:B:m:W:P,A:256:64,16:0,32",
            ),
            (
                "gpgpu_cache:dl2_texture_only",
                "0",
            ),
            (
                "gpgpu_dram_partition_queues",
                "32:32:32:32",
            ),
            (
                "gpgpu_perf_sim_memcpy",
                "1",
            ),
            (
                "gpgpu_memory_partition_indexing",
                "4",
            ),
            (
                "gpgpu_cache:il1",
                "N:8:128:4,L:R:f:N:L,S:2:48,4",
            ),
            (
                "gpgpu_inst_fetch_throughput",
                "8",
            ),
            (
                "gpgpu_tex_cache:l1",
                "N:16:128:24,L:R:m:N:L,T:128:4,128:2",
            ),
                    (
                        "gpgpu_const_cache:l1",
                "N:128:64:2,L:R:f:N:L,S:2:64,4",
            ),
            (
                "gpgpu_perfect_inst_const_cache",
                "1",
            ),
            (
                "network_mode",
                "1",
            ),
            (
                "inter_config_file",
                "config_pascal_islip.icnt",
            ),
            (
                "gpgpu_l2_rop_latency",
                "120",
            ),
            (
                "dram_latency",
                "100",
            ),
            (
                "gpgpu_dram_scheduler",
                "1",
            ),
            (
                "gpgpu_frfcfs_dram_sched_queue_size",
                "64",
            ),
            (
                "gpgpu_dram_return_queue_size",
                "64",
            ),
            (
                "gpgpu_n_mem_per_ctrlr",
                "1",
            ),
            (
                "gpgpu_dram_buswidth",
                "4",
            ),
            (
                "gpgpu_dram_burst_length",
                "8",
            ),
                    (
                "dram_data_command_freq_ratio",
                "4",
            ),
            (
                "gpgpu_mem_address_mask",
                "1",
            ),
            (
                "gpgpu_mem_addr_mapping",
                "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS",
            ),
            (
                "gpgpu_dram_timing_opt",
                "\"nbk=16:CCD=2:RRD=8:RCD=16:RAS=37:RP=16:RC=52:\n                        CL=16:WL=6:CDLR=7:WR=16:nbkgrp=4:CCDL=4:RTPL=3\"",
            ),
            (
                "dram_bnk_indexing_policy",
                "0",
            ),
            (
                "dram_bnkgrp_indexing_policy",
                "1",
            ),
            (
                "gpgpu_memlatency_stat",
                "14",
            ),
            (
                "gpgpu_runtime_stat",
                "500",
            ),
            (
                "enable_ptx_file_line_stats",
                "1",
            ),
            (
                "visualizer_enabled",
                "0",
            ),
        ];
        diff::assert_eq!(have: have, want: want);
        Ok(())
    }
}
