use clap::Parser;

#[derive(Parser, Debug, Clone, PartialEq, Eq)]
#[clap()]
pub struct TraceConfig {
    #[clap(
        long = "trace",
        help = "traces kernel file traces kernel file directory",
        default_value = "./traces/kernelslist.g"
    )]
    pub traces_filename: String,
    #[clap(
        long = "trace_opcode_latency_initiation_int",
        help = "Opcode latencies and initiation for integers in trace driven mode <latency,initiation>",
        default_value = "4,1"
    )]
    pub trace_opcode_latency_initiation_int: String,
    #[clap(
        long = "trace_opcode_latency_initiation_sp",
        help = "Opcode latencies and initiation for sp in trace driven mode <latency,initiation>",
        default_value = "4,1"
    )]
    pub trace_opcode_latency_initiation_sp: String,
    #[clap(
        long = "trace_opcode_latency_initiation_dp",
        help = "Opcode latencies and initiation for dp in trace driven mode <latency,initiation>",
        default_value = "4,1"
    )]
    pub trace_opcode_latency_initiation_dp: String,
    #[clap(
        long = "trace_opcode_latency_initiation_sfu",
        help = "Opcode latencies and initiation for sfu in trace driven mode <latency,initiation>",
        default_value = "4,1"
    )]
    pub trace_opcode_latency_initiation_sfu: String,
    #[clap(
        long = "trace_opcode_latency_initiation_tensor",
        help = "Opcode latencies and initiation for tensor in trace driven mode <latency,initiation>",
        default_value = "4,1"
    )]
    pub trace_opcode_latency_initiation_tensor: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_1",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_1: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_2",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_2: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_3",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_3: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_4",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_4: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_5",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_5: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_6",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_6: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_7",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_7: String,
    #[clap(
        long = "trace_opcode_latency_initiation_spec_op_8",
        help = "specialized unit config <latency,initiation>",
        default_value = "4,4"
    )]
    pub trace_opcode_latency_initiation_spec_op_8: String,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            traces_filename: "./traces/kernelslist.g".to_string(),
            trace_opcode_latency_initiation_int: "4,1".to_string(),
            trace_opcode_latency_initiation_sp: "4,1".to_string(),
            trace_opcode_latency_initiation_dp: "4,1".to_string(),
            trace_opcode_latency_initiation_sfu: "4,1".to_string(),
            trace_opcode_latency_initiation_tensor: "4,1".to_string(),
            trace_opcode_latency_initiation_spec_op_1: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_2: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_3: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_4: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_5: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_6: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_7: "4,4".to_string(),
            trace_opcode_latency_initiation_spec_op_8: "4,4".to_string(),
        }
    }
}
