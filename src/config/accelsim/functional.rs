use super::Boolean;
use clap::Parser;

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[clap(
    // trailing_var_arg = true,
    // allow_hyphen_values = true,
    // arg_required_else_help = false
)]
pub struct FunctionalConfig {
    #[clap(
        long = "gpgpu_ptx_use_cuobjdump",
        help = "Use cuobjdump to extract ptx and sass from binaries",
        // value_parser = super::BoolParser{},
        default_value = "1"
    )]
    pub m_ptx_use_cuobjdump: Boolean,
    #[clap(
        long = "gpgpu_experimental_lib_support",
        help = "Try to extract code from cuda libraries [Broken because of unknown cudaGetExportTable]",
        // value_parser = super::BoolParser{},
        default_value = "0"
    )]
    pub m_experimental_lib_support: Boolean,
    #[clap(
        long = "checkpoint_option",
        help = "checkpointing flag (0 = no checkpoint)",
        default_value = "0"
    )]
    pub checkpoint_option: u32,
    #[clap(
        long = "checkpoint_kernel",
        help = "checkpointing during execution of which kernel (1- 1st kernel)",
        default_value = "1"
    )]
    pub checkpoint_kernel: u32,
    #[clap(
        long = "checkpoint_CTA",
        help = "checkpointing after # of CTA (< less than total CTA)",
        default_value = "0"
    )]
    pub checkpoint_cta: u32,
    #[clap(
        long = "resume_option",
        help = "resume flag (0 = no resume)",
        default_value = "0"
    )]
    pub resume_option: u32,
    #[clap(
        long = "resume_kernel",
        help = "Resume from which kernel (1= 1st kernel)",
        default_value = "0"
    )]
    pub resume_kernel: u32,
    #[clap(
        long = "resume_CTA",
        help = "resume from which CTA",
        default_value = "0"
    )]
    pub resume_cta: u32,
    #[clap(
        long = "checkpoint_CTA_t",
        help = "resume from which CTA",
        default_value = "0"
    )]
    pub checkpoint_cta_t: u32,
    #[clap(
        long = "checkpoint_insn_Y",
        help = "resume from which CTA",
        default_value = "0"
    )]
    pub checkpoint_insn_y: u32,
    #[clap(
        long = "gpgpu_ptx_convert_to_ptxplus",
        help = "Convert SASS (native ISA) to ptxplus and run ptxplus",
        default_value = "0"
    )]
    pub m_ptx_convert_to_ptxplus: Boolean,
    #[clap(
        long = "gpgpu_ptx_force_max_capability",
        help = "Force maximum compute capability",
        default_value = "0"
    )]
    pub m_ptx_force_max_capability: u32,
    #[clap(
        long = "gpgpu_ptx_inst_debug_to_file",
        help = "Dump executed instructions' debug information to file",
        // value_parser = super::BoolParser{},
        default_value = "0"
    )]
    pub g_ptx_inst_debug_to_file: Boolean,
    #[clap(
        long = "gpgpu_ptx_inst_debug_file",
        help = "Executed instructions' debug output file",
        default_value = "inst_debug.txt"
    )]
    pub g_ptx_inst_debug_file: String,
    #[clap(
        long = "gpgpu_ptx_inst_debug_thread_uid",
        help = "Thread UID for executed instructions' debug output",
        default_value = "1"
    )]
    pub g_ptx_inst_debug_thread_uid: u32,
}

impl Default for FunctionalConfig {
    fn default() -> Self {
        Self {
            m_ptx_use_cuobjdump: true.into(),
            m_experimental_lib_support: false.into(),
            checkpoint_option: 0,
            checkpoint_kernel: 1,
            checkpoint_cta: 0,
            resume_option: 0,
            resume_kernel: 0,
            resume_cta: 0,
            checkpoint_cta_t: 0,
            checkpoint_insn_y: 0,
            m_ptx_convert_to_ptxplus: false.into(),
            m_ptx_force_max_capability: 0,
            g_ptx_inst_debug_to_file: false.into(),
            g_ptx_inst_debug_file: "inst_debug.txt".to_string(),
            g_ptx_inst_debug_thread_uid: 1,
        }
    }
}
