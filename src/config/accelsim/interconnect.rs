use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InterconnectConfig {
    #[clap(
        long = "network_mode",
        help = "Interconnection network mode",
        default_value = "1"
    )]
    pub g_network_mode: u32,
    #[clap(
        long = "inter_config_file",
        help = "Interconnection network config file",
        default_value = "mesh"
    )]
    pub g_network_config_filename: String,
    // parameters for local xbar
    #[clap(
        long = "icnt_in_buffer_limit",
        help = "in_buffer_limit",
        default_value = "64"
    )]
    pub in_buffer_limit: u32,
    #[clap(
        long = "icnt_out_buffer_limit",
        help = "out_buffer_limit",
        default_value = "64"
    )]
    pub out_buffer_limit: u32,
    #[clap(long = "icnt_subnets", help = "subnets", default_value = "2")]
    pub subnets: u32,
    #[clap(long = "icnt_arbiter_algo", help = "arbiter_algo", default_value = "1")]
    pub arbiter_algo: u32,
    #[clap(long = "icnt_verbose", help = "icnt_verbose", default_value = "0")]
    pub verbose: u32,
    #[clap(long = "icnt_grant_cycles", help = "grant_cycles", default_value = "1")]
    pub grant_cycles: u32,
}

impl Default for InterconnectConfig {
    fn default() -> Self {
        Self {
            g_network_mode: 1,
            g_network_config_filename: "mesh".to_string(),
            in_buffer_limit: 64,
            out_buffer_limit: 64,
            subnets: 2,
            arbiter_algo: 1,
            verbose: 0,
            grant_cycles: 1,
        }
    }
}
