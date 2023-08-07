use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scheduler {
    pub num_single_issue: u64,
    pub num_dual_issue: u64,
    pub issue_raw_hazard_stall: u64,
    pub issue_control_hazard_stall: u64,
    pub issue_pipeline_stall: u64,
}
