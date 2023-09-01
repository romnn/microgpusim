use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[clap()]
pub struct TimingConfig {
    #[clap(long = "nbk", help = "number of banks")]
    pub nbk: Option<u32>,
    #[clap(long = "CCD", help = "column to column delay")]
    pub t_ccd: Option<u32>,
    #[clap(
        long = "RRD",
        help = "minimal delay between activation of rows in different banks"
    )]
    pub t_rrd: Option<u32>,
    #[clap(long = "RCD", help = "row to column delay")]
    pub t_rdc: Option<u32>,
    #[clap(long = "RAS", help = "time needed to activate row")]
    pub t_ras: Option<u32>,
    #[clap(long = "RP", help = "time needed to precharge (deactivate) row")]
    pub t_rp: Option<u32>,
    #[clap(long = "RC", help = "row cycle time")]
    pub t_rc: Option<u32>,
    #[clap(long = "CDLR", help = "switching from write to read (changes tWTR)")]
    pub t_cdlr: Option<u32>,
    #[clap(long = "WR", help = "last data-in to row precharge")]
    pub t_wr: Option<u32>,
    #[clap(long = "CL", help = "CAS latency")]
    pub cl: Option<u32>,
    #[clap(long = "WL", help = "Write latency")]
    pub wl: Option<u32>,
    #[clap(long = "nbkgrp", help = "number of bank groups", default_value = "1")]
    pub nbkgrp: u32,
    #[clap(
        long = "CCDL",
        help = "column to column delay between accesses to different bank groups",
        default_value = "0"
    )]
    pub t_ccdl: u32,
    #[clap(
        long = "RTPL",
        help = "read to precharge delay between accesses to different bank groups",
        default_value = "0"
    )]
    pub t_rtpl: u32,
}

impl Default for TimingConfig {
    fn default() -> Self {
        Self {
            nbk: None,
            t_ccd: None,
            t_rrd: None,
            t_rdc: None,
            t_ras: None,
            t_rp: None,
            t_rc: None,
            t_cdlr: None,
            t_wr: None,
            cl: None,
            wl: None,
            nbkgrp: 1,
            t_ccdl: 0,
            t_rtpl: 0,
        }
    }
}
