use clap::Parser;
use color_eyre::eyre;

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[clap(
    trailing_var_arg = true,
    allow_hyphen_values = true,
    arg_required_else_help = false
)]
pub struct Config {
    #[clap(short, help = "todo")]
    pub gpgpu_ptx_instruction_classification: Option<usize>,
    // #[clap(long = "kernels", help = "path to kernelslist.g file")]
    // pub kernelslist: Option<PathBuf>,
    //
    // #[clap(flatten)]
    // pub sim_config: SimConfig,
    //
    // #[clap(long = "log-file", help = "write simuation output to log file")]
    // pub log_file: Option<PathBuf>,
    //
    // #[clap(long = "stats-file", help = "parse simulation stats into csv file")]
    // pub stats_file: Option<PathBuf>,
    //
    // #[clap(
    //     long = "timeout",
    //     help = "timeout",
    //     value_parser = parse_duration_string,
    // )]
    // pub timeout: Option<Duration>,
    // Command line to start child process
    // #[clap(short, multiple_values = true, allow_hyphen_values = true)]
    // #[clap(short, last = true, allow_hyphen_values = true)]
    // #[clap(last = true, allow_hyphen_values = true)]
    // #[clap(last = true, allow_hyphen_values = true)]
    #[clap(num_args(0..), allow_hyphen_values = true)]
    unknown: Vec<String>,
}

use once_cell::sync::Lazy;
use regex::Regex;

static ARGUMENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    let arg = r"(-[\w\d\-:]+)";
    let single_quoted_string = "(?:'(?:[^\\']|\\.)*')";
    let double_quoted_string = r#"(?:"(?:[^\"]|\\.)*")"#;
    let value_excluding_comment = r"(?:[^#\n]+)";
    let trailing_comment = r"(?:#.*)?";
    let pattern = [
        r"^\s*",
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
    dbg!(&pattern);
    regex::RegexBuilder::new(&pattern)
        .multi_line(true)
        .build()
        .unwrap()
});

pub fn extract_arguments<'a>(config: &'a str) -> impl Iterator<Item = (&'a str, &'a str)> + '_ {
    ARGUMENT_REGEX.captures_iter(config).filter_map(|cap| {
        let key = cap.get(1)?.as_str().trim();
        let value = cap.get(2)?.as_str().trim();
        Some((key, value))
    })
}

impl Config {
    pub fn from_config_str(config: impl AsRef<str>) -> eyre::Result<Self> {
        let args = extract_arguments(config.as_ref()).flat_map(|(key, value)| [key, value]);
        let args: Vec<&str> = args.collect();
        dbg!(&args);
        let config = Self::try_parse_from(&args)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use color_eyre::eyre;
    use similar_asserts as diff;
    use std::path::PathBuf;

    #[test]
    fn test_read_config_file_gtx1080() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let config = std::fs::read_to_string(config_path)?;
        diff::assert_eq!(
            super::Config::from_config_str(config)?,
            super::Config {
                gpgpu_ptx_instruction_classification: Some(0),
                unknown: vec![],
            },
        );
        Ok(())
    }

    #[test]
    fn test_extract_arguments() -> eyre::Result<()> {
        let manifest_dir = PathBuf::from(std::env!("CARGO_MANIFEST_DIR"));
        let config_path = manifest_dir.join("accelsim/gtx1080/gpgpusim.config");
        let config = std::fs::read_to_string(config_path)?;
        let arguments: Vec<_> = super::extract_arguments(&config).collect();
        let expected = vec![
            ("-gpgpu_ptx_instruction_classification", "0"),
            ("-gpgpu_ptx_sim_mode", "0"),
            ("-gpgpu_ptx_force_max_capability", "60"),
            ("-gpgpu_ptx_convert_to_ptxplus", "0"),
            ("-gpgpu_ptx_save_converted_ptxplus", "0"),
            ("-gpgpu_n_clusters", "20"),
            ("-gpgpu_n_cores_per_cluster", "1"),
            ("-gpgpu_n_mem", "8"),
            ("-gpgpu_n_sub_partition_per_mchannel", "2"),
            ("-gpgpu_clock_domains", "1607.0:1607.0:1607.0:2500.0"),
            ("-gpgpu_shader_registers", "65536"),
            ("-gpgpu_occupancy_sm_number", "60"),
            ("-gpgpu_shader_core_pipeline", "2048:32"),
            ("-gpgpu_shader_cta", "32"),
            ("-gpgpu_simd_model", "1"),
            ("-gpgpu_pipeline_widths", "4,0,0,1,1,4,0,0,1,1,6"),
            ("-gpgpu_num_sp_units", "4"),
            ("-gpgpu_num_sfu_units", "1"),
            ("-gpgpu_tensor_core_avail", "0"),
            ("-gpgpu_num_tensor_core_units", "0"),
            ("-ptx_opcode_latency_int", "4,13,4,5,145"),
            ("-ptx_opcode_initiation_int", "1,2,2,2,8"),
            ("-ptx_opcode_latency_fp", "4,13,4,5,39"),
            ("-ptx_opcode_initiation_fp", "1,2,1,1,4"),
            ("-ptx_opcode_latency_dp", "8,19,8,8,330"),
            ("-ptx_opcode_initiation_dp", "1,2,1,1,130"),
            ("-gpgpu_cache:dl1", "N:64:128:6,L:L:m:N:H,A:128:8,8"),
            ("-gpgpu_shmem_size", "98304"),
            ("-gpgpu_gmem_skip_L1D", "1"),
            (
                "-gpgpu_cache:dl2",
                "N:64:128:16,L:B:m:W:L,A:1024:1024,4:0,32",
            ),
            ("-gpgpu_cache:dl2_texture_only", "0"),
            ("-gpgpu_cache:il1", "N:8:128:4,L:R:f:N:L,A:2:48,4"),
            ("-gpgpu_tex_cache:l1", "N:16:128:24,L:R:m:N:L,F:128:4,128:2"),
            ("-gpgpu_const_cache:l1", "N:128:64:2,L:R:f:N:L,A:2:64,4"),
            ("-gpgpu_operand_collector_num_units_sp", "20"),
            ("-gpgpu_operand_collector_num_units_sfu", "4"),
            ("-gpgpu_operand_collector_num_units_mem", "8"),
            ("-gpgpu_operand_collector_num_in_ports_sp", "4"),
            ("-gpgpu_operand_collector_num_out_ports_sp", "4"),
            ("-gpgpu_operand_collector_num_in_ports_sfu", "1"),
            ("-gpgpu_operand_collector_num_out_ports_sfu", "1"),
            ("-gpgpu_operand_collector_num_in_ports_mem", "1"),
            ("-gpgpu_operand_collector_num_out_ports_mem", "1"),
            ("-gpgpu_num_reg_banks", "32"),
            ("-gpgpu_shmem_num_banks", "32"),
            ("-gpgpu_shmem_limited_broadcast", "0"),
            ("-gpgpu_shmem_warp_parts", "1"),
            ("-gpgpu_max_insn_issue_per_warp", "2"),
            ("-network_mode", "1"),
            ("-inter_config_file", "config_fermi_islip.icnt"),
            ("-gpgpu_l2_rop_latency", "120"),
            ("-dram_latency", "100"),
            ("-gpgpu_dram_scheduler", "1"),
            ("-gpgpu_frfcfs_dram_sched_queue_size", "64"),
            ("-gpgpu_dram_return_queue_size", "116"),
            ("-gpgpu_n_mem_per_ctrlr", "1"),
            ("-gpgpu_dram_buswidth", "4"),
            ("-gpgpu_dram_burst_length", "8"),
            ("-dram_data_command_freq_ratio", "4"),
            ("-gpgpu_mem_address_mask", "1"),
            (
                "-gpgpu_mem_addr_mapping",
                "dramid@8;00000000.00000000.00000000.00000000.0000RRRR.RRRRRRRR.RBBBCCCC.BCCSSSSS",
            ),
            (
                "-gpgpu_dram_timing_opt",
                r#""nbk=16:CCD=2:RRD=6:RCD=12:RAS=28:RP=12:RC=40:
                        CL=12:WL=4:CDLR=5:WR=12:nbkgrp=1:CCDL=0:RTPL=0""#,
            ),
            ("-gpgpu_num_sched_per_core", "2"),
            ("-gpgpu_scheduler", "gto"),
            ("-gpgpu_memlatency_stat", "14"),
            ("-gpgpu_runtime_stat", "500"),
            ("-enable_ptx_file_line_stats", "1"),
            ("-visualizer_enabled", "0"),
            ("-power_simulation_enabled", "0"),
        ];
        diff::assert_eq!(have: arguments, want: expected);
        Ok(())
    }
}
