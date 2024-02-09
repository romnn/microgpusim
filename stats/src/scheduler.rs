use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scheduler {
    pub execution_unit_issue: HashMap<String, u64>,
    pub num_single_issue: u64,
    pub num_dual_issue: u64,
    pub issue_raw_hazard_stall: u64,
    pub issue_control_hazard_stall: u64,
    pub issue_pipeline_stall: u64,
}

impl std::ops::AddAssign for Scheduler {
    fn add_assign(&mut self, other: Self) {
        for (k, v) in other.execution_unit_issue {
            *self.execution_unit_issue.entry(k).or_insert(0) += v;
        }
        self.num_single_issue += other.num_single_issue;
        self.num_dual_issue += other.num_dual_issue;
        self.issue_raw_hazard_stall += other.issue_raw_hazard_stall;
        self.issue_control_hazard_stall += other.issue_control_hazard_stall;
        self.issue_pipeline_stall += other.issue_pipeline_stall;
    }
}
