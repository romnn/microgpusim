use super::Boolean;
use clap::Parser;

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[clap()]
pub struct PTXConfig {
    #[clap(
        long = "save_embedded_ptx",
        help = "saves ptx files embedded in binary as <n>.ptx",
        // value_parser = super::BoolParser{},
        default_value = "0",
    )]
    pub g_save_embedded_ptx: Boolean,
    #[clap(
        long = "keep",
        help = "keep intermediate files created by GPGPU-Sim when interfacing with external programs",
        // value_parser = super::BoolParser{},
        default_value = "0",
    )]
    pub keep: Boolean,
    #[clap(
        long = "gpgpu_ptx_save_converted_ptxplus",
        help = "Saved converted ptxplus to a file",
        // value_parser = super::BoolParser{},
        default_value = "0"
    )]
    pub g_ptx_save_converted_ptxplus: Boolean,
    #[clap(
        long = "gpgpu_occupancy_sm_number",
        help = "The SM number to pass to ptxas when getting register usage for computing GPU occupancy. This parameter is required in the config.",
        default_value = "0"
    )]
    pub g_occupancy_sm_number: u32,
    #[clap(
        long = "ptx_opcode_latency_int",
        help = "Opcode latencies for integers <ADD,MAX,MUL,MAD,DIV,SHFL> Default 1,1,19,25,145,32",
        default_value = "1,1,19,25,145,32"
    )]
    pub opcode_latency_int: String,
    #[clap(
        long = "ptx_opcode_latency_fp",
        help = "Opcode latencies for single precision floating points <ADD,MAX,MUL,MAD,DIV> Default 1,1,1,1,30",
        default_value = "1,1,1,1,30"
    )]
    pub opcode_latency_fp: String,
    #[clap(
        long = "ptx_opcode_latency_dp",
        help = "Opcode latencies for double precision floating points <ADD,MAX,MUL,MAD,DIV> Default 8,8,8,8,335",
        default_value = "8,8,8,8,335"
    )]
    pub opcode_latency_dp: String,
    #[clap(
        long = "ptx_opcode_latency_sfu",
        help = "Opcode latencies for SFU instructions Default 8",
        default_value = "8"
    )]
    pub opcode_latency_sfu: String,
    #[clap(
        long = "ptx_opcode_latency_tesnor",
        help = "Opcode latencies for Tensor instructions Default 64",
        default_value = "64"
    )]
    pub opcode_latency_tensor: String,
    #[clap(
        long = "ptx_opcode_initiation_int",
        help = "Opcode initiation intervals for integers <ADD,MAX,MUL,MAD,DIV,SHFL> Default 1,1,4,4,32,4",
        default_value = "1,1,4,4,32,4"
    )]
    pub opcode_initiation_int: String,
    #[clap(
        long = "ptx_opcode_initiation_fp",
        help = "Opcode initiation intervals for single precision floating points <ADD,MAX,MUL,MAD,DIV> Default 1,1,1,1,5",
        default_value = "1,1,1,1,5"
    )]
    pub opcode_initiation_fp: String,
    #[clap(
        long = "ptx_opcode_initiation_dp",
        help = "Opcode initiation intervals for double precision floating points <ADD,MAX,MUL,MAD,DIV> Default 8,8,8,8,130",
        default_value = "8,8,8,8,130"
    )]
    pub opcode_initiation_dp: String,
    #[clap(
        long = "ptx_opcode_initiation_sfu",
        help = "Opcode initiation intervals for sfu instructions Default 8",
        default_value = "8"
    )]
    pub opcode_initiation_sfu: String,
    #[clap(
        long = "ptx_opcode_initiation_tensor",
        help = "Opcode initiation intervals for tensor instructions Default 64",
        default_value = "64"
    )]
    pub opcode_initiation_tensor: String,
    #[clap(
        long = "cdp_latency",
        help = "CDP API latency <cudaStreamCreateWithFlags, cudaGetParameterBufferV2_init_perWarp, cudaGetParameterBufferV2_perKernel, cudaLaunchDeviceV2_init_perWarp, cudaLaunchDevicV2_perKernel>. Default 7200,8000,100,12000,1600",
        default_value = "7200,8000,100,12000,1600"
    )]
    pub cdp_latency_str: String,
}

impl Default for PTXConfig {
    fn default() -> Self {
        Self {
            g_save_embedded_ptx: false.into(),
            keep: false.into(),
            g_ptx_save_converted_ptxplus: false.into(),
            g_occupancy_sm_number: 0,
            opcode_latency_int: "1,1,19,25,145,32".to_string(),
            opcode_latency_fp: "1,1,1,1,30".to_string(),
            opcode_latency_dp: "8,8,8,8,335".to_string(),
            opcode_latency_sfu: "8".to_string(),
            opcode_latency_tensor: "64".to_string(),
            opcode_initiation_int: "1,1,4,4,32,4".to_string(),
            opcode_initiation_fp: "1,1,1,1,5".to_string(),
            opcode_initiation_dp: "8,8,8,8,130".to_string(),
            opcode_initiation_sfu: "8".to_string(),
            opcode_initiation_tensor: "64".to_string(),
            cdp_latency_str: "7200,8000,100,12000,1600".to_string(),
        }
    }
}
