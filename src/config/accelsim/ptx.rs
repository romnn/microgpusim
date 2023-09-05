use super::Boolean;
use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[clap()]
pub struct PTXConfig {
    #[clap(
        long = "save_embedded_ptx",
        help = "saves ptx files embedded in binary as <n>.ptx",
        default_value = "0"
    )]
    pub g_save_embedded_ptx: Boolean,
    #[clap(
        long = "keep",
        help = "keep intermediate files created by GPGPU-Sim when interfacing with external programs",
        default_value = "0"
    )]
    pub keep: Boolean,
    #[clap(
        long = "gpgpu_ptx_save_converted_ptxplus",
        help = "Saved converted ptxplus to a file",
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

#[derive(Default, Debug, Clone, Hash, PartialEq, Eq)]
pub struct OpcodeLatencies {
    pub add_sub: Option<u64>,
    pub max_min: Option<u64>,
    pub mul: Option<u64>,
    pub mad: Option<u64>,
    pub div: Option<u64>,
    pub shuffle: Option<u64>,
}

impl OpcodeLatencies {
    pub fn max(&self) -> Option<u64> {
        [
            self.add_sub,
            self.max_min,
            self.mul,
            self.mad,
            self.div,
            self.shuffle,
        ]
        .iter()
        .filter_map(Option::as_ref)
        .max()
        .copied()
    }
}

fn parse_latencies(config: &str) -> Result<OpcodeLatencies, std::num::ParseIntError> {
    use itertools::Itertools;
    let latencies: Vec<u64> = config
        .split(',')
        .map(str::trim)
        .map(str::parse)
        .try_collect()?;
    Ok(OpcodeLatencies {
        add_sub: latencies.first().copied(),
        max_min: latencies.get(1).copied(),
        mul: latencies.get(2).copied(),
        mad: latencies.get(3).copied(),
        div: latencies.get(4).copied(),
        shuffle: latencies.get(5).copied(),
    })
}

impl PTXConfig {
    pub fn int_latencies(&self) -> Result<OpcodeLatencies, std::num::ParseIntError> {
        parse_latencies(&self.opcode_latency_int)
    }

    pub fn sp_latencies(&self) -> Result<OpcodeLatencies, std::num::ParseIntError> {
        parse_latencies(&self.opcode_latency_fp)
    }

    pub fn dp_latencies(&self) -> Result<OpcodeLatencies, std::num::ParseIntError> {
        parse_latencies(&self.opcode_latency_dp)
    }
}

#[cfg(test)]
mod tests {
    use color_eyre::eyre;
    use utils::diff;

    #[test]
    fn test_opcode_latencies_int() -> eyre::Result<()> {
        let int_latencies = super::parse_latencies("1,1,19,25,145,32")?;
        assert_eq!(int_latencies.max(), Some(145));
        diff::assert_eq!(
            have: int_latencies,
            want: super::OpcodeLatencies {
                add_sub: Some(1),
                max_min: Some(1),
                mul: Some(19),
                mad: Some(25),
                div: Some(145),
                shuffle: Some(32),
            }
        );
        Ok(())
    }

    #[test]
    fn test_opcode_latencies_dp() -> eyre::Result<()> {
        let dp_latencies = super::parse_latencies("8,8,8,8,335")?;
        assert_eq!(dp_latencies.max(), Some(335));
        diff::assert_eq!(
            have: dp_latencies,
            want: super::OpcodeLatencies {
                add_sub: Some(8),
                max_min: Some(8),
                mul: Some(8),
                mad: Some(8),
                div: Some(335),
                shuffle: None,
            }
        );
        Ok(())
    }

    #[test]
    fn test_opcode_latencies_sp() -> eyre::Result<()> {
        let sp_latencies = super::parse_latencies("1,1,1,1,30")?;
        assert_eq!(sp_latencies.max(), Some(30));
        diff::assert_eq!(
            have: sp_latencies,
            want: super::OpcodeLatencies {
                add_sub: Some(1),
                max_min: Some(1),
                mul: Some(1),
                mad: Some(1),
                div: Some(30),
                shuffle: None,
            }
        );
        Ok(())
    }
}
