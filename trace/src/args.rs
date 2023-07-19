use bitvec::{array::BitArray, field::BitField, BitArr};

use crate::common;
use crate::instrumentor::Instrumentor;

#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Default, Clone)]
pub struct Args {
    pub instr_data_width: u32,
    pub instr_opcode_id: std::ffi::c_int,
    /// instruction offset is equivalent to virtual pc
    pub instr_offset: u32,
    pub instr_idx: u32,
    pub instr_predicate_num: std::ffi::c_int,
    pub instr_predicate_is_neg: bool,
    pub instr_predicate_is_uniform: bool,
    pub instr_mem_space: u8,
    pub instr_is_mem: bool,
    pub instr_is_load: bool,
    pub instr_is_store: bool,
    pub instr_is_extended: bool,
    // mem addr
    #[allow(dead_code)]
    pub mref_idx: u64,
    // register info
    pub dest_reg: Option<u32>,
    // num_dest_regs: u32,
    // dest_regs: [u32; common::MAX_DST as usize],
    pub num_src_regs: u32,
    pub src_regs: [u32; common::MAX_SRC as usize],
    // receiver channel
    pub ptr_channel_dev: u64,
    pub line_num: u32,
}

impl Args {
    pub fn instrument(&self, trace_ctx: &Instrumentor<'_>, instr: &mut nvbit_rs::Instruction<'_>) {
        instr.add_call_arg_guard_pred_val();
        instr.add_call_arg_const_val32(self.instr_data_width);
        instr.add_call_arg_const_val32(self.instr_opcode_id.try_into().unwrap_or_default());
        instr.add_call_arg_const_val32(self.instr_offset);
        instr.add_call_arg_const_val32(self.instr_idx);
        instr.add_call_arg_const_val32(self.line_num);

        instr.add_call_arg_const_val32(self.instr_mem_space.into());
        instr.add_call_arg_const_val32(self.instr_predicate_num.try_into().unwrap_or_default());

        // pack binary flags due to 11 argument limitation
        let mut flags: BitArr!(for 32) = BitArray::ZERO;
        flags.set(0, self.instr_is_mem);
        flags.set(1, self.instr_is_load);
        flags.set(2, self.instr_is_store);
        flags.set(3, self.instr_is_extended);
        flags.set(4, self.instr_predicate_is_neg);
        flags.set(5, self.instr_predicate_is_uniform);
        instr.add_call_arg_const_val32(flags.load_be::<u32>());

        // register info is allocated on the device and passed by pointer
        let reg_info = common::reg_info_t {
            // has_dest_reg: self.dest_reg.is_some(),
            // dest_reg: self.dest_reg.unwrap_or(0),
            dest_regs: [self.dest_reg.unwrap_or(0)],
            num_dest_regs: u32::from(self.dest_reg.is_some()),
            src_regs: self.src_regs,
            num_src_regs: self.num_src_regs,
        };
        let dev_reg_info = unsafe { common::allocate_reg_info(reg_info) };
        instr.add_call_arg_const_val64(dev_reg_info as u64);
        trace_ctx.defer_free_device_memory(dev_reg_info as u64);

        // memory reference 64 bit address
        if self.instr_is_mem {
            instr.add_call_arg_mref_addr64(0);
        } else {
            instr.add_call_arg_const_val64(u64::MAX);
        }

        // pointer to device channel for sending packets
        instr.add_call_arg_const_val64(self.ptr_channel_dev);

        // add "space" for kernel_id function pointer,
        // that will be set at launch time
        // (64 bit value at offset 0 of the dynamic arguments)
        instr.add_call_arg_launch_val64(0);
    }
}
