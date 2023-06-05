#include "opcode.hpp"

const char *g_opcode_string[] = {
    "ABS_OP",      "ADD_OP",       "ADDP_OP",       "ADDC_OP",      "AND_OP",
    "ANDN_OP",     "ATOM_OP",      "BAR_OP",        "BFE_OP",       "BFI_OP",
    "BFIND_OP",    "BRA_OP",       "BRX_OP",        "BREV_OP",      "BRKPT_OP",
    "MMA_OP",      "MMA_LD_OP",    "MMA_ST_OP",     "CALL_OP",      "CALLP_OP",
    "CLZ_OP",      "CNOT_OP",      "COS_OP",        "CVT_OP",       "CVTA_OP",
    "DIV_OP",      "DP4A_OP",      "EX2_OP",        "EXIT_OP",      "FMA_OP",
    "ISSPACEP_OP", "LD_OP",        "LDU_OP",        "LG2_OP",       "MAD24_OP",
    "MAD_OP",      "MADC_OP",      "MADP_OP",       "MAX_OP",       "MEMBAR_OP",
    "MIN_OP",      "MOV_OP",       "MUL24_OP",      "MUL_OP",       "NEG_OP",
    "NANDN_OP",    "NORN_OP",      "NOT_OP",        "OR_OP",        "ORN_OP",
    "PMEVENT_OP",  "POPC_OP",      "PREFETCH_OP",   "PREFETCHU_OP", "PRMT_OP",
    "RCP_OP",      "RED_OP",       "REM_OP",        "RET_OP",       "RETP_OP",
    "RSQRT_OP",    "SAD_OP",       "SELP_OP",       "SETP_OP",      "SET_OP",
    "SHFL_OP",     "SHL_OP",       "SHR_OP",        "SIN_OP",       "SLCT_OP",
    "SQRT_OP",     "SST_OP",       "SSY_OP",        "ST_OP",        "SUB_OP",
    "SUBC_OP",     "SULD_OP",      "SURED_OP",      "SUST_OP",      "SUQ_OP",
    "TEX_OP",      "TRAP_OP",      "VABSDIFF_OP",   "VADD_OP",      "VMAD_OP",
    "VMAX_OP",     "VMIN_OP",      "VSET_OP",       "VSHL_OP",      "VSHR_OP",
    "VSUB_OP",     "VOTE_OP",      "ACTIVEMASK_OP", "XOR_OP",       "NOP_OP",
    "BREAK_OP",    "BREAKADDR_OP",
};
