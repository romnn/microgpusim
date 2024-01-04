import os
import typing
from typing import Sequence
import click
import tempfile
import glob
from tqdm import tqdm
import time
import bitarray
import bitarray.util
import pandas as pd
from collections import defaultdict
import sympy as sym
import pycryptosat
from pathlib import Path
from pprint import pprint
from wasabi import color
from cuda import cuda, nvrtc
import numpy as np
import sys
import itertools

import gpucachesim.cmd as cmd_utils
from gpucachesim import REPO_ROOT_DIR
import gpucachesim.utils as utils

sys.path.insert(0, str(REPO_ROOT_DIR / "CuAssembler"))
import CuAsm as asm

# from pynvrtc.interface import NVRTCInterface
# inter = NVRTCInterface("/usr/local/cuda-11.8/lib64/libnvrtc.so")
# inter = nvrtc.Interface(Path(os.environ["CUDA_HOME"] or "/usr/local/cuda") / "lib64/libnvrtc.so")

# from CuAssembler import CuAsm as asm

CUDA_SAMPLES = REPO_ROOT_DIR / "test-apps/cuda-samples-11.8/"
SASS_CODE_REPO = REPO_ROOT_DIR / "plot/asm/"


def code_repo(arch):
    return SASS_CODE_REPO / "{}.repo.txt".format(arch)


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


def make(c, arches, j=4, target=""):
    cmd = [
        "make",
        "-B",
        "-j",
        str(j),
        "-C",
        str(c),
    ]
    if target != "":
        cmd.append(target)
    cmd = " ".join(cmd)
    print(cmd)
    sms = " ".join([str(int(arch.lower().removeprefix("sm_"))) for arch in arches])
    _, stdout, stderr, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=30 * 60,
        env={**os.environ, **{"SMS": sms}},
    )


def get_sass(executable, arch="sm_61"):
    cmd = ["cuobjdump", "-sass", "-arch", str(arch), str(executable)]
    cmd = " ".join(cmd)
    print(cmd)
    _, stdout, _, _ = cmd_utils.run_cmd(
        cmd,
        timeout_sec=30 * 60,
    )
    return stdout


def compile_cuda(code, arch="sm_61"):
    # compile nvcc *.cu -o test
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        code_path = temp_dir / "code.cu"
        with open(str(code_path), "w") as f:
            f.write(code)

        cubin_path = temp_dir / "out.cubin"

        cmd = ["nvcc", str(code_path), "-o", str(cubin_path)]
        cmd = " ".join(cmd)
        print(cmd)
        _, stdout, _, _ = cmd_utils.run_cmd(
            cmd,
            timeout_sec=30 * 60,
        )
        return stdout


@main.command()
@click.option(
    "--compile",
    "compile",
    type=bool,
    is_flag=True,
    help="compile CUDA sample applications",
)
@click.option(
    "--fail-fast",
    "fail_fast",
    type=bool,
    is_flag=True,
    help="fail fast",
)
@click.option(
    "--arch",
    "arch",
    type=str,
    default="sm_61",
    help="sm architecture (e.g. sm_75)",
)
def dump_sass(compile, fail_fast, arch):
    # sudo apt-get install freeglut3 freeglut3-dev
    # make -j -C $CUDA_SAMPLES SMS="61"
    # print(CUDA_SAMPLES)
    # print(arch)
    # executables = list(
    #     [
    #         f
    #         for f in CUDA_SAMPLES.glob("**/*")
    #         if f.suffix == "" and f.is_file() and os.access(str(f.absolute()), os.X_OK)
    #     ]
    # )
    # CUDA_SAMPLES / "Samples/2_Concepts_and_Techniques/"
    # sample_apps = sorted(list(CUDA_SAMPLES.rglob("*/*/")))
    if compile:
        sample_apps = sorted(
            list(
                [
                    Path(f)
                    for f in glob.glob(str(CUDA_SAMPLES.absolute()) + "/*/*/*")
                    if Path(f).is_dir()
                ]
            )
        )
        # pprint(sample_apps)
        for sample_app_dir in sample_apps:
            rel_path = sample_app_dir.relative_to(CUDA_SAMPLES)
            makefile = sample_app_dir / "Makefile"
            if not makefile.is_file():
                continue
            print("compiling {}".format(color(rel_path, fg="cyan")))
            try:
                make(c=sample_app_dir, arches=[arch], target="clean")
                make(c=sample_app_dir, arches=[arch])
            except cmd_utils.ExecStatusError as err:
                print("{: >10}: {}".format(str(color("ERROR", fg="red")), err))
                if fail_fast:
                    raise err
            except cmd_utils.ExecTimeoutError as err:
                print("{: >10}: {}".format(str(color("TIMEOUT", fg="yellow")), err))
                if fail_fast:
                    raise err

    # dump sass for all binaries
    bin_dir = CUDA_SAMPLES / "bin/x86_64/linux/release/"
    executables = sorted(list(bin_dir.iterdir()))
    # pprint(executables)
    for executable in executables:
        try:
            sass = get_sass(executable, arch=arch)
        except cmd_utils.ExecStatusError as err:
            print("{: >10}: {}".format(str(color("ERROR", fg="red")), err))
            if fail_fast:
                raise err
            else:
                continue
        except cmd_utils.ExecTimeoutError as err:
            print("{: >10}: {}".format(str(color("TIMEOUT", fg="yellow")), err))
            if fail_fast:
                raise err
            else:
                continue

        # print(sass)
        sass_path = SASS_CODE_REPO / arch / "{}.{}.sass".format(executable.name, arch)
        sass_path.parent.mkdir(parents=True, exist_ok=True)

        print(
            "wrote SASS for {} to {}".format(
                executable.name, color(sass_path, fg="cyan")
            )
        )
        with open(sass_path, "w+") as f:
            f.write(sass)


@main.command()
@click.option(
    "--arch",
    "arch",
    type=str,
    default="sm_61",
    help="sm architecture (e.g. sm_75)",
)
@click.option(
    "--limit",
    type=int,
    help="limit the number of sass files to use for the repo",
)
def build_repo(arch, limit):
    sass_files = sorted(list((SASS_CODE_REPO / arch).iterdir()))
    if limit is not None:
        sass_files = sass_files[: int(limit)]

    repos = asm.CuInsAssemblerRepos(arch=arch)  # initialize an empty repository
    for sass_file in sass_files:
        with open(sass_file, "r") as f:
            feeder = asm.CuInsFeeder(f, archfilter=arch)
            # update the repo with instructions from feeder
            repos.update(feeder)

    # save the repo
    repos.save2file(code_repo(arch))


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            _, err_string = cuda.cudaGetErrorString(err)
            raise RuntimeError(
                "Cuda Error ({}): {}".format(err, err_string.decode("utf-8"))
            )
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            _, err_string = nvrtc.nvrtcGetErrorString(err)
            raise RuntimeError(
                "Nvrtc Error ({}): {}".format(err, err_string.decode("utf-8"))
            )
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


import re


def get_ptx_from_cuda_program(prog) -> str:
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    ASSERT_DRV(err)
    ptx = b" " * ptxSize
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    ASSERT_DRV(err)
    return ptx.decode("utf-8")


def get_cubin_from_cuda_program(prog) -> bytes:
    """Get CUBIN from compilation

    Note: CUBIN is only available if we use an actual architecture (sm_61)
    instead of a virtual one only (compute_61)
    """
    err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)

    ASSERT_DRV(err)
    print(cubinSize)
    cubin = b" " * cubinSize
    (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
    ASSERT_DRV(err)
    return cubin


@main.command()
@click.option(
    "--arch",
    "arch",
    type=str,
    default="sm_61",
    help="sm architecture (e.g. sm_75)",
)
def instruction_latency(arch):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        code = """
        extern "C" __global__
        void saxpy(float a, float *x, float *y, float *out, size_t n)
        {
         size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         if (tid < n) {
           out[tid] = a * x[tid] + y[tid];
         }
        }
        """

        kernel_name = "inst_latency"
        code = 'extern "C" __global__ void {}'.format(kernel_name)
        code += "(int a, int b, unsigned int *result, unsigned int *time)"
        code += """
        {
          // int a = 582712;
          // int b = 2783829;

          unsigned int start_time = clock();
          long prod = a * b;
          unsigned int end_time = clock();
          *result = prod;
          *time = (end_time - start_time);
        }
        """
        print(code)

        # Create program
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(code), b"saxpy.cu", 0, [], [])
        ASSERT_DRV(err)

        # Compile program
        opts = [
            "--gpu-architecture={}".format(arch).encode("utf-8"),
            b"--dopt=on",
            # b'--ptxas-options="-O3"',
            # b"--ptxas-options=-O3",
        ]
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        ASSERT_DRV(err)

        cubin = get_cubin_from_cuda_program(prog)
        # print(cubin)

        cubin_path = temp_dir / "out.cubin"
        with open(cubin_path, "wb") as f:
            f.write(cubin)

        cuasm_path = temp_dir / "out.cuasm"
        cb = asm.CubinFile(cubin_path)
        cb.saveAsCuAsm(cuasm_path)

        with open(cuasm_path, "r") as f:
            cuasm = f.read()

        # print(cuasm)

        # get the code
        kernel_asm_regex = re.compile(
            r"("
            + kernel_name
            + r":\n\s*\.text\."
            + kernel_name
            + r":(\n.*)*END of sections)",
            re.MULTILINE,
        )
        kernel_asm = kernel_asm_regex.findall(cuasm)[0][0]
        print(kernel_asm)


def np_ptr(v):
    return v.ctypes.data


@main.command()
@click.option(
    "--arch",
    "arch",
    type=str,
    default="sm_61",
    help="sm architecture (e.g. sm_75)",
)
def saxpy(arch):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        code = """
    extern "C" __global__
    void saxpy(float a, float *x, float *y, float *out, size_t n)
    {
     size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid < n) {
       out[tid] = a * x[tid] + y[tid];
     }
    }
    """
        # /usr/local/cuda/lib64/libnvrtc.so
        # Create program
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(code), b"saxpy.cu", 0, [], [])
        ASSERT_DRV(err)

        # Compile program
        opts = [
            # b"-Xptxas=dlcm=ca",
            b"--dopt=on",
            "--gpu-architecture={}".format(arch).encode("utf-8"),
            # b"-dlcm=ca",
            # b'-Xptxas="-O3 -dlcm=ca"',
            # b'-Xptxas="-O3 -dlcm=ca"',
        ]
        print(opts)
        (err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
        ASSERT_DRV(err)

        # Get PTX from compilation
        ptx = get_ptx_from_cuda_program(prog)
        print(ptx)
        # err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        # ASSERT_DRV(err)
        # ptx = b" " * ptxSize
        # (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
        # ASSERT_DRV(err)
        # print(ptx.decode("utf-8"))

        # Get CUBIN from compilation
        # Note: CUBIN is only available if we use an actual architecture (sm_61)
        # instead of a virtual one only (compute_61)
        # err, cubinSize = nvrtc.nvrtcGetCUBINSize(prog)
        # ASSERT_DRV(err)
        # print(cubinSize)
        # cubin = b" " * cubinSize
        # (err,) = nvrtc.nvrtcGetCUBIN(prog, cubin)
        # ASSERT_DRV(err)

        cubin = get_cubin_from_cuda_program(prog)
        print(cubin)

        cubin_path = temp_dir / "out.cubin"
        with open(cubin_path, "wb") as f:
            f.write(cubin)

        cuasm_path = temp_dir / "out.cuasm"
        cb = asm.CubinFile(cubin_path)
        cb.saveAsCuAsm(cuasm_path)

        with open(cuasm_path, "r") as f:
            cuasm = f.read()

        print(cuasm)

        # def __parseKernelText(self, section, line_start, line_end):
        # line = self.__mLines[lineidx]
        #
        # nline = CuAsmParser.stripComments(line).strip()
        # self.__mLineNo = lineidx + 1
        #
        # if len(nline)==0 or (self.m_label.match(nline) is not None) or (self.m_directive.match(nline) is not None):
        #     continue
        #
        # res = p_textline.match(nline)
        # if res is None:
        #     self.__assert(False, 'Unrecognized code text!')
        #
        # ccode_s = res.groups()[0]
        # icode_s = res.groups()[1]
        #
        # if c_ControlCodesPattern.match(ccode_s) is None:
        #     self.__assert(False, f'Illegal control code text "{ccode_s}"!')
        #
        # addr = self.m_Arch.getInsOffsetFromIndex(ins_idx)
        # c_icode_s = self.__evalInstructionFixup(section, addr, icode_s)
        #
        # #print("Parsing %s : %s"%(ccode_s, c_icode_s))
        # try:
        #     kasm.push(addr, c_icode_s, ccode_s)
        # except Exception as e:
        #     self.__assert(False, 'Error when assembling instruction "%s":\n        %s'%(nline, e))
        #
        # ins_idx += 1

        # rewrite text sections
        # codebytes = kasm.genCode()

        # new_cuasm_path = temp_dir / "modified.cuasm"
        # with open(new_cuasm_path, "w") as f:
        #     f.write(new_cuasm)

        new_cubin_path = temp_dir / "modified.cubin"
        assembler = asm.CuAsmParser()
        assert code_repo(arch).is_file()
        assembler.setInsAsmRepos(str(code_repo(arch)), arch=arch)
        assembler.parse(cuasm_path)
        assembler.saveAsCubin(str(new_cubin_path))

        # Initialize CUDA Driver API
        (err,) = cuda.cuInit(0)
        ASSERT_DRV(err)

        # Retrieve handle for device 0
        err, cuDevice = cuda.cuDeviceGet(0)
        ASSERT_DRV(err)

        # Create context
        err, context = cuda.cuCtxCreate(0, cuDevice)
        ASSERT_DRV(err)

        NUM_THREADS = 512  # Threads per block
        NUM_BLOCKS = 32768  # Blocks per grid

        a = np.array([2.0], dtype=np.float32)
        n = np.array(NUM_THREADS * NUM_BLOCKS, dtype=np.uint32)
        buffer_size = n * a.itemsize

        h_x = np.random.rand(n).astype(dtype=np.float32)
        h_y = np.random.rand(n).astype(dtype=np.float32)
        h_out = np.zeros(n).astype(dtype=np.float32)

        err, d_x_ptr = cuda.cuMemAlloc(buffer_size)
        ASSERT_DRV(err)
        err, d_y_ptr = cuda.cuMemAlloc(buffer_size)
        ASSERT_DRV(err)
        err, d_out_ptr = cuda.cuMemAlloc(buffer_size)
        ASSERT_DRV(err)

        (err,) = cuda.cuMemcpyHtoD(d_x_ptr, np_ptr(h_x), buffer_size)
        ASSERT_DRV(err)
        (err,) = cuda.cuMemcpyHtoD(d_y_ptr, np_ptr(h_y), buffer_size)
        ASSERT_DRV(err)

        d_x = np.array([int(d_x_ptr)], dtype=np.uint64)
        d_y = np.array([int(d_y_ptr)], dtype=np.uint64)
        d_out = np.array([int(d_out_ptr)], dtype=np.uint64)

        kernel_args = [a, d_x, d_y, d_out, n]
        kernel_args = np.array([np_ptr(arg) for arg in kernel_args], dtype=np.uint64)

        def launch_sass_kernel(
            cubin: typing.Union[bytes, os.PathLike],
            kernel_name: bytes,
            grid_x: int,
            grid_y: int,
            grid_z: int,
            block_x: int,
            block_y: int,
            block_z: int,
            dynamic_shared_memory: int,
            stream: int,
            kernel_args: Sequence[int],
        ):
            if isinstance(cubin, str) or isinstance(cubin, Path):
                # load from file
                err, module = cuda.cuModuleLoad(str(cubin))
            elif isinstance(cubin, bytes):
                # load from data
                err, module = cuda.cuModuleLoadData(cubin)
            else:
                raise TypeError(f"cannot load module from cubin type {type(cubin)}")

            ASSERT_DRV(err)

            err, kernel = cuda.cuModuleGetFunction(module, bytes(kernel_name))
            ASSERT_DRV(err)

            (err,) = cuda.cuLaunchKernel(
                kernel,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                dynamic_shared_memory,
                stream,
                kernel_args,
                0,  # extra (ignore)
            )
            return err

        err = launch_sass_kernel(
            cubin=cubin,
            kernel_name=b"saxpy",  # must match the name of the kernel in the cubin!
            grid_x=NUM_BLOCKS,
            grid_y=1,
            grid_z=1,
            block_x=NUM_THREADS,
            block_y=1,
            block_z=1,
            dynamic_shared_memory=0,
            stream=0,
            kernel_args=kernel_args.ctypes.data,
        )
        ASSERT_DRV(err)

        (err,) = cuda.cuMemcpyDtoH(np_ptr(h_out), d_out_ptr, buffer_size)
        ASSERT_DRV(err)

        # Assert values are same after running kernel
        h_z = a * h_x + h_y
        if not np.allclose(h_out, h_z):
            raise ValueError("Error outside tolerance for host-device vectors")
        else:
            print("VALID!!!")


def get_bit(i, n):
    return (n & (1 << i)) >> i


def bitstring(n, num_bits):
    return "".join([str(b) for b in bits(n, num_bits)])


def bits(n, num_bits):
    return [get_bit(int(i), int(n)) for i in reversed(range(num_bits))]


def count_clauses(eq) -> int:
    if isinstance(eq, (sym.logic.boolalg.And, sym.logic.boolalg.Or)):
        return len(eq.args)
    elif isinstance(eq, sym.logic.Not):
        return count_clauses(eq.args[0])
    elif isinstance(eq, sym.logic.Symbol):
        return 1
    else:
        raise ValueError("cannot count clauses of {}".format(eq))


def count_symbols(eq: typing.Any) -> int:
    vars = set()
    _get_symbols(vars, eq)
    return len(vars)


def _get_symbols(vars: typing.Set[str], eq: typing.Any):  # -> typing.Set[str]:
    if isinstance(eq, (sym.logic.boolalg.And, sym.logic.boolalg.Or)):
        # print(eq.args)
        # return vars.update(utils.flatten([
        # vars.update(utils.flatten([
        # list(_get_symbols(set(), a)) for a in eq.args]))
        for a in eq.args:
            _get_symbols(vars, a)
    elif isinstance(eq, sym.logic.Not):
        # return vars.update(list(_get_symbols(set(), eq.args[0])))
        # vars.update(list(_get_symbols(set(), eq.args[0])))
        _get_symbols(vars, eq.args[0])
    elif isinstance(eq, sym.Symbol):
        # return set([str(eq)])
        vars.update([str(eq)])
    else:
        raise ValueError("cannot count clauses of {}".format(eq))


def solve_mapping_table(df, num_bits=64, use_and=False):
    df = df.astype(int)
    sets = df["set"].unique()
    num_sets = len(sets)
    print("num sets", num_sets)
    num_set_bits = int(np.log2(num_sets))
    print("num set bits", num_set_bits)
    assert sets.max() < 2**num_set_bits

    sols = defaultdict(list)
    for output_set_bit in range(num_set_bits):

        def build_eq(addr):
            gates = []
            for bit in range(num_bits):
                toggle = sym.symbols(f"b{bit}")
                gate = sym.logic.boolalg.And(toggle, bool(get_bit(int(bit), int(addr))))
                gates.append(gate)

            if use_and:
                return sym.logic.boolalg.And(*gates)
            else:
                return sym.logic.boolalg.Or(*gates)

        equations = []
        for addr, set_id in df.to_numpy():
            eq = build_eq(addr)
            if not bool(get_bit(output_set_bit, set_id)):
                eq = ~eq
            # print(eq)
            equations.append(eq)

        unknown_vars = [sym.symbols(f"b{bit}") for bit in range(num_bits)]

        print(
            "solving {} equations with {} unknown variables for set bit {}".format(
                len(equations), len(unknown_vars), output_set_bit
            )
        )

        # if False:
        #     default_sol = {v: False for v in unknown_vars}
        #     for eq in equations:
        #         if eq == True or eq == False:
        #             continue
        #
        #         right_sol = {
        #             **default_sol,
        #             **{sym.symbols(f"b7_1"): True, sym.symbols(f"b13_2"): True},
        #         }
        #         # print(right_sol)
        #         right = eq.subs(right_sol)
        #         if right == True:
        #             continue
        #         print("\n\n")
        #         print(eq)
        #         print("right", right)
        #         wrong_sol = {
        #             **default_sol,
        #             **{sym.symbols(f"b10_1"): True, sym.symbols(f"b15_2"): True},
        #         }
        #         wrong = eq.subs(wrong_sol)
        #         # print("wrong", wrong)
        #         # assert right
        #         # assert not wrong

        all_models = True
        per_bit_solutions = list(
            sym.satisfiable(sym.logic.boolalg.And(*equations), all_models=all_models)
        )
        for sol_num, sol in enumerate(per_bit_solutions):
            if isinstance(sol, dict):
                for symbol, enabled in sol.items():
                    if enabled:
                        print(symbol, "=", enabled)
            elif isinstance(sol, bool):
                pass
            if sol == False:
                print(
                    color(
                        "NO solution {} for set bit {}".format(sol_num, output_set_bit),
                        fg="red",
                    )
                )
            else:
                print(
                    color(
                        "FOUND solution {} for set bit {}: {}".format(
                            sol_num, output_set_bit, sol
                        ),
                        fg="green",
                    )
                )
            sols[output_set_bit].append(sol)

    return sols


def sympy_to_pycrypto_cnf(equations):
    """Convert sympy to pycryptosat CNF clauses.

    Warning: this is astronomically slow for XOR.
    XOR is converted using cut and convert and results in exponential explosion of clauses.
    Even without simplification, generating a single XOR CNF with degree 8 can take more than 15 seconds.
    """
    symbol_mapping = dict()
    clauses = []
    xor_clauses = []
    for eq in tqdm(equations, desc="rewriting as CNF", disable=False):
        if eq == True or eq == False:
            continue

        def get_var_id(var):
            var = str(var)
            negated = var.startswith("~")
            var = var.removeprefix("~")
            var_id = symbol_mapping.get(var)
            if var_id is None:
                var_id = len(symbol_mapping) + 1
                symbol_mapping[var] = var_id

            var_id *= -1 if negated else 1
            return var_id

        # print(eq)
        # print(type(eq))

        if isinstance(eq, DummyXor):
            vars = []
            for var in eq.vars:
                var_id = get_var_id(var)
                vars.append(var_id)
            xor_clauses.append((vars, eq.rhs))
            continue

        eq = sym.logic.boolalg.to_cnf(eq, simplify=False, force=False)
        # print(len(eq.args))

        # if isinstance(eq, sym.Symbol):
        #     continue

        # ignore free standing symbols
        # if not isinstance(eq, sym.logic.boolalg.And):
        #     continue

        def is_free_standing_symbol(expr) -> bool:
            if isinstance(expr, sym.Symbol):
                return True
            if isinstance(expr, sym.Not):
                # print(expr.args)
                if len(expr.args) == 1 and isinstance(expr.args[0], sym.Symbol):
                    return True
                # if all([isinstance(arg, sym.Symbol) for arg in expr.args]):
                #     return True
            return False

        def convert_or(expr):
            new_clause = []
            for var in expr.args:
                new_clause.append(convert_free_standing_var(var))
                # var_id = get_var_id(var)
                # new_clause.append(var_id)
            return new_clause

        def convert_free_standing_var(expr):
            return get_var_id(expr)

        if isinstance(eq, sym.logic.boolalg.And):
            for clause in eq.args:
                # new_clause = []
                if isinstance(clause, sym.logic.boolalg.Or):
                    clauses.append(convert_or(clause))
                #     for var in clause.args:
                #         var_id = get_var_id(var)
                #         # print(var, var_id)
                #         new_clause.append(var_id)
                #
                #     # print(new_clause)
                #     clauses.append(new_clause)
                elif is_free_standing_symbol(clause):
                    clauses.append([convert_free_standing_var(clause)])
                else:
                    raise TypeError(
                        f"unexpected expression {clause} of type {type(clause)}"
                    )

        elif isinstance(eq, sym.logic.boolalg.Or):
            clauses.append(convert_or(eq))

        elif is_free_standing_symbol(eq):
            clauses.append([convert_free_standing_var(eq)])
        else:
            raise TypeError(f"unexpected expression {eq} of type {type(eq)}")

    return (clauses, xor_clauses), symbol_mapping


def solve_mapping_table_xor(
    df, num_bits=64, degree=2, use_bits=None, all_models=False, use_sympy=False
):
    sets = list(df["set"].unique())
    num_sets = len(sets)
    print("num sets", num_sets)
    num_set_bits = int(np.log2(num_sets))
    print("num set bits", num_set_bits)

    assert num_sets > 1
    assert num_set_bits > 0

    sols = defaultdict(list)
    for output_set_bit in range(num_set_bits):

        def build_eq(addr):
            if isinstance(use_bits, list) or isinstance(use_bits, tuple):
                bits = []
                for bit in use_bits:
                    toggle = sym.symbols(f"b{bit}")
                    gate = sym.logic.boolalg.And(
                        toggle, bool(get_bit(int(bit), int(addr)))
                    )
                    bits.append(gate)

                return sym.logic.boolalg.Xor(*bits)
            else:
                degrees = []
                for deg in range(degree):
                    bits = []
                    for bit in range(num_bits):
                        toggle = sym.symbols(f"b{bit}_{deg}")
                        gate = sym.logic.boolalg.And(
                            toggle, bool(get_bit(int(bit), int(addr)))
                        )
                        bits.append(gate)

                    degrees.append(sym.logic.boolalg.Or(*bits))

                return sym.logic.boolalg.Xor(*degrees)

        equations = []
        for addr, set in tqdm(df.to_numpy(), desc="generate equations"):
            eq = build_eq(addr)
            # if isinstance(eq, sym.Symbol):
            #     raise ValueError
            rhs = bool(get_bit(output_set_bit, set))
            print(eq, "==", rhs)
            if not rhs:
                eq = ~eq
            equations.append(eq)
            # if len(equations) > 30:
            #     break

        unknown_vars = []
        if isinstance(use_bits, list):
            unknown_vars += [sym.symbols(f"b{bit}") for bit in use_bits]
        else:
            for deg in range(degree):
                unknown_vars += [
                    sym.symbols(f"b{bit}_{deg}") for bit in range(num_bits)
                ]

        # if False:
        #     default_sol = {v: False for v in unknown_vars}
        #     for eq in equations:
        #         if eq == True or eq == False:
        #             continue
        #
        #         # right = sym.simplify(eq.subs({b[7]: 1, b[13]: 1}))
        #         # wrong = sym.simplify(eq.subs({b[7]: 2, b[13]: 1}))
        #         right_sol = {
        #             **default_sol,
        #             **{sym.symbols(f"b7_1"): True, sym.symbols(f"b13_2"): True},
        #         }
        #         # print(right_sol)
        #         right = eq.subs(right_sol)
        #         if right == True:
        #             continue
        #         print("\n\n")
        #         print(eq)
        #         print("right", right)
        #         wrong_sol = {
        #             **default_sol,
        #             **{sym.symbols(f"b10_1"): True, sym.symbols(f"b15_2"): True},
        #         }
        #         wrong = eq.subs(wrong_sol)
        #         # print("wrong", wrong)
        #         # assert right
        #         # assert not wrong

        print(
            "solving {} equations with {} unknown variables for set bit {}".format(
                len(equations), len(unknown_vars), output_set_bit
            )
        )

        # for eq in equations:
        #     print(eq)
        #     sol = sym.satisfiable(eq)
        #     print(sol)

        if use_sympy:
            start = time.time()
            print("sympy: start solving")
            per_bit_sols = list(
                sym.satisfiable(
                    sym.logic.boolalg.And(*equations), all_models=all_models
                )
            )
            for sol_num, sol in enumerate(per_bit_sols):
                if isinstance(sol, dict):
                    for symbol, enabled in sol.items():
                        if enabled:
                            print(symbol, "=", enabled)
                elif isinstance(sol, bool):
                    pass
                # pprint(dict(sol))
                # for s in dict(sol):
                #     pprint(dict(s))
                # sol = sym.solve(equations, unknown_vars, dict=True)
                # sol = sym.solveset(equations, unknown_vars, domain=sym.FiniteSet(0, 1))
                if sol == False:
                    print(
                        color(
                            "NO solution {} for set bit {}".format(
                                sol_num, output_set_bit, sol
                            ),
                            fg="green",
                        )
                    )
                else:
                    print(
                        color(
                            "FOUND solution {} for set bit {}: {}".format(
                                sol_num, output_set_bit, sol
                            ),
                            fg="red",
                        )
                    )
                sols[output_set_bit].append(sol)

            print("sympy: completed in {:8.2f} sec".format(time.time() - start))
        else:
            print("pycryptosat: rewriting equations as CNF")
            (clauses, _), symbol_mapping = sympy_to_pycrypto_cnf(equations)

            # for clause in clauses:
            #     print(clause)
            #     assert isinstance(clause, list)

            # pprint(clauses)
            # pprint(symbol_mapping)
            start = time.time()
            print("pycryptosat: start solving")
            s = pycryptosat.Solver()
            s.add_clauses(clauses)
            sat, solution_values = s.solve()
            if not sat:
                print(
                    color("NO solution for set bit {}".format(output_set_bit), fg="red")
                )
            else:
                sol = dict()
                for var, var_id in symbol_mapping.items():
                    if solution_values[var_id]:
                        sol[var] = True

                print(
                    color(
                        "FOUND solution for set bit {}: {}".format(output_set_bit, sol),
                        fg="green",
                    )
                )
                sols[output_set_bit].append(sol)

            print("pycryptosat: completed in {:8.2f} sec".format(time.time() - start))

    return sols


def bool_equal(a, b):
    both_true = sym.logic.boolalg.And(a, b)
    both_false = sym.logic.boolalg.And(~a, ~b)
    return both_true | both_false


class DummyXor:
    def __init__(self, vars, rhs):
        self.vars = vars
        self.rhs = rhs


def solve_mapping_table_xor_fast(df, num_bits=64, degree=2):
    sets = list(df["set"].unique())
    num_sets = len(sets)
    print("num sets", num_sets)
    num_set_bits = int(np.log2(num_sets))
    print("num set bits", num_set_bits)

    assert num_sets > 1
    assert num_set_bits > 0

    sols = defaultdict(list)
    for output_set_bit in range(num_set_bits):

        def build_eq(addr):
            # generate the bits for each degree
            variable_declarations = []
            for deg in range(degree):
                toggles = []
                bits = []
                for bit in range(num_bits):
                    toggle = sym.symbols(f"b{bit}_{deg}", bit=bit, deg=deg)
                    gate = sym.logic.boolalg.And(
                        toggle, bool(get_bit(int(bit), int(addr)))
                    )
                    if bool(get_bit(int(bit), int(addr))):
                        toggles.append(str(toggle))
                    bits.append(gate)

                dest = "d_" + "_".join(toggles)
                variable_declarations.append(
                    (sym.symbols(dest), sym.logic.boolalg.Or(*bits))
                )
                # degrees.append(bool_eq(sym.symbols(f"d{deg}"), sym.logic.boolalg.Or(*bits)))
                # degrees.append([b+1 for b in range(num_bits) if bool(get_bit(int(b), int(addr))])

            return variable_declarations
            # return sym.logic.boolalg.Xor(*degrees)

        # equations = []
        equations = []
        for addr, set_id in tqdm(df.to_numpy(), desc="generate equations"):
            vars = []
            for var, eq in build_eq(addr):
                # declare that var == eq
                if eq == True or eq == False:
                    continue
                # print(var, "==", eq)
                equations.append(bool_equal(var, eq))

                # if not bool(get_bit(output_set_bit, set_id)):
                #     eq = ~eq
                # bool(get_bit(output_set_bit, set_id))
                # equations.append(eq)

            # declare that XOR(*vars) == bool(get_bit(output_set_bit, set_id))
            equations.append(DummyXor(vars, bool(get_bit(output_set_bit, set_id))))

            # if len(equations) > 30:
            #     break

        # pprint(equations)
        (clauses, xor_clauses), symbol_mapping = sympy_to_pycrypto_cnf(equations)
        # pprint(xor_clauses)
        # pprint(clauses)
        # pprint(symbol_mapping)
        #
        # return

        # print(
        #     "solving {} equations with {} unknown variables for set bit {}".format(
        #         len(equations), len(unknown_vars), output_set_bit
        #     )
        # )

        # print("pycryptosat: rewriting equations as CNF")
        # clauses, symbol_mapping = sympy_to_pycrypto_cnf(equations)

        print("pycryptosat: start solving")
        start = time.time()

        s = pycryptosat.Solver()
        s.add_clauses(clauses)

        # add the xor clauses
        for vars, rhs in xor_clauses:
            s.add_xor_clause(vars, rhs=rhs)

        # s.add_xor_clause([symbol_mapping[f"d{deg}"] for deg in range(degree)], rhs=True)

        sat, solution_values = s.solve()
        if not sat:
            print(color("NO solution for set bit {}".format(output_set_bit), fg="red"))
        else:
            sol = dict()
            for var, var_id in symbol_mapping.items():
                if solution_values[var_id]:
                    sol[var] = True

            print(
                color(
                    "FOUND solution for set bit {}: {}".format(output_set_bit, sol),
                    fg="green",
                )
            )
            sols[output_set_bit].append(sol)

        print("pycryptosat: completed in {:8.2f} sec".format(time.time() - start))


@main.command()
def solve_simple():
    num_bits = 3

    def test_set_index_function(addr: int) -> int:
        set_bit = 2
        return (addr & (1 << set_bit)) >> set_bit

    # build table
    data = []
    for n in range(2**num_bits):
        # print("n", bits(n, num_bits), "set", set)
        data.append([n, test_set_index_function(n)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(data, columns=["n", "set"])
    print(df)
    print(compute_set_probability(df))

    sols = solve_mapping_table(df, num_bits=num_bits)
    # print(sols)


@main.command()
def solve_bitwise_hash():
    line_size = 128
    line_size_log2 = int(np.log2(line_size))  # 7
    num_sets = 4
    num_sets_log2 = int(np.log2(num_sets))  # 2

    num_bits = 20

    def test_set_index_function(addr: int) -> int:
        higher_bits = addr >> (line_size_log2 + num_sets_log2)
        index = (addr >> line_size_log2) & (num_sets - 1)
        return (index) ^ (higher_bits & (num_sets - 1))

    # build table
    data = []
    for line in range((2**num_bits) // line_size):
        addr = line * line_size
        # print("n", bits(n, num_bits), "set", set)
        data.append([addr, test_set_index_function(addr)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(data, columns=["n", "set"])
    print(df)
    print(compute_set_probability(df))

    # s = pycryptosat.Solver()
    # # x1 xor x2
    # s.add_xor_clause(xor_clause=[1, 2], rhs=True)
    # sat, solution = s.solve()
    # print("is satisfiable:", sat)
    # print("solution:", solution[1:])

    # expect: set[0] = 7 xor 9
    # expect: set[2] = 8 xor 10
    # sols = solve_mapping_table_xor(df, num_bits=num_bits)
    sols = solve_mapping_table_xor_fast(df, num_bits=num_bits)
    # print(sols)


@main.command()
def solve_ipoly_hash():
    line_size = 128
    line_size_log2 = int(np.log2(line_size))  # 7
    num_sets = 16  # only 16, 32, or 64 supported
    num_sets_log2 = int(np.log2(num_sets))  # 4

    num_bits = 20

    def test_set_index_function(addr: int, endian="little") -> int:
        higher_bits = addr >> (line_size_log2 + num_sets_log2)
        index = (addr >> line_size_log2) & (num_sets - 1)
        # print(np.binary_repr(num_sets - 1))
        higher_bits = bitarray.util.int2ba(higher_bits, length=64, endian=endian)
        index = bitarray.util.int2ba(index, length=64, endian=endian)
        # print("higher bits", higher_bits)
        # print("index", index)

        # higher_bits = np.binary_repr(higher_bits, width=64)
        # index = np.binary_repr(index, width=64)
        # print(higher_bits)
        # print(index)
        #
        # higher_bits = bitarray(higher_bits)
        # index = bitarray(index)

        # index = bitarray((addr >> line_size_log2) & (num_sets - 1))
        new_index = bitarray.util.int2ba(0, length=4, endian=endian)

        # print((
        #     higher_bits[11]
        #     , higher_bits[10]
        #     , higher_bits[9]
        #     , higher_bits[8]
        #     , higher_bits[6]
        #     , higher_bits[4]
        #     , higher_bits[3]
        #     , higher_bits[0]
        #     , index[0]
        # ))

        # set bit 0 uses 9 bits
        new_index[0] = (
            higher_bits[11]
            ^ higher_bits[10]
            ^ higher_bits[9]
            ^ higher_bits[8]
            ^ higher_bits[6]
            ^ higher_bits[4]
            ^ higher_bits[3]
            ^ higher_bits[0]
            ^ index[0]
        )

        # set bit 1 uses 9 bits
        new_index[1] = (
            higher_bits[12]
            ^ higher_bits[8]
            ^ higher_bits[7]
            ^ higher_bits[6]
            ^ higher_bits[5]
            ^ higher_bits[3]
            ^ higher_bits[1]
            ^ higher_bits[0]
            ^ index[1]
        )

        # set bit 2 uses 8 bits
        new_index[2] = (
            higher_bits[9]
            ^ higher_bits[8]
            ^ higher_bits[7]
            ^ higher_bits[6]
            ^ higher_bits[4]
            ^ higher_bits[2]
            ^ higher_bits[1]
            ^ index[2]
        )

        # set bit 3 uses 8 bits
        new_index[3] = (
            higher_bits[10]
            ^ higher_bits[9]
            ^ higher_bits[8]
            ^ higher_bits[7]
            ^ higher_bits[5]
            ^ higher_bits[3]
            ^ higher_bits[2]
            ^ index[3]
        )

        # print(new_index)
        return bitarray.util.ba2int(new_index)

    # build table
    data = []
    for line in range((2**num_bits) // line_size):
        addr = line * line_size
        print("n", bits(line, num_bits), "set", test_set_index_function(addr))
        data.append([addr, test_set_index_function(addr)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(data, columns=["n", "set"])
    print(df)
    print(compute_set_probability(df))

    # only solve a mapping for set=0
    # df = df[df["set"] == 0]
    # set_mask = df["set"] == 0
    # df.loc[set_mask, "set"] = 1
    # df.loc[~set_mask, "set"] = 0

    if False:
        deg1 = sym.symbols("b1_1") | sym.symbols("b2_1") | sym.symbols("b3_1")
        deg2 = sym.symbols("b1_2") | sym.symbols("b2_2") | sym.symbols("b3_2")
        deg3 = sym.symbols("b1_3") | sym.symbols("b2_3") | sym.symbols("b3_3")

        # deg1 = sym.symbols("b1_1") | sym.symbols("b2_1")
        # deg2 = sym.symbols("b1_2") | sym.symbols("b2_2")
        # deg3 = sym.symbols("b1_3") | sym.symbols("b2_3")

        # deg1 = sym.symbols("b1_1")
        # deg2 = sym.symbols("b1_2")
        # deg3 = sym.symbols("b1_3")

        eq = deg1 ^ deg2 ^ deg3
        print(eq)
        pprint(eq.args)
        num_xors = len(eq.args)
        print(num_xors)
        num_ors = len(eq.args[0].args)
        num_ors = max(1, num_ors - 1)
        print(num_ors)
        eq = sym.logic.boolalg.to_cnf(eq, simplify=False, force=False)
        pprint(eq.args)
        print(len(eq.args))
        # print((2**(num_xors - 1)))
        print((2 ** (num_xors - 1)) * 2 ** (num_ors) + (num_ors))
        # print((2**(num_xors - 1)) * (num_ors + 1) + 1)

    if False:
        higher_bits_start = line_size_log2 + num_sets_log2
        index_bits_start = line_size_log2
        bit_combination = [
            higher_bits_start + 11,
            higher_bits_start + 10,
            higher_bits_start + 9,
            higher_bits_start + 8,
            higher_bits_start + 6,
            higher_bits_start + 4,
            higher_bits_start + 3,
            higher_bits_start + 0,
            index_bits_start + 0,
        ]
        pprint(bit_combination)
        sols = solve_mapping_table_xor(df, num_bits=num_bits, use_bits=bit_combination)
        return

    for degree in range(8, 10):
        bit_combinations = list(
            itertools.combinations(
                [line_size_log2 + bit for bit in range(num_bits - line_size_log2)],
                degree,
            )
        )
        print(bit_combinations[0])
        for bit_combination in bit_combinations:
            assert len(bit_combination) == degree
            print(f"==== SOLVING XOR [degree={degree}, bits={bit_combination}]")
            sols = solve_mapping_table_xor(
                df, num_bits=num_bits, use_bits=bit_combination
            )
    # print(sols)


def compute_set_probability(df):
    set_probability = df["set"].value_counts().reset_index()
    set_probability["prob"] = set_probability["count"] / set_probability["count"].sum()
    return set_probability


@main.command()
def solve_linear():
    line_size = 128
    line_size_log2 = int(np.log2(line_size))  # 7
    num_bits = 20

    num_sets = 4

    def test_set_index_function(addr: int) -> int:
        return (addr >> line_size_log2) & (num_sets - 1)
        # set_bit = 2
        # return (addr & (1 << set_bit)) >> set_bit

    # build table
    data = []
    for line in range((2**num_bits) // line_size):
        addr = line * line_size
        # print("n", bits(n, num_bits), "set", set)
        data.append([addr, test_set_index_function(addr)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(data, columns=["n", "set"])
    print(df)
    print(compute_set_probability(df))

    # expect:
    # set index 0: b[7]
    # set index 1: b[8]
    sols = solve_mapping_table(df, use_and=False, num_bits=num_bits)
    # print(sols)


@main.command()
def solve_fermi():
    num_bits = 20
    line_size = 128
    line_size_log2 = int(np.log2(line_size))
    print("line size (log2)", line_size_log2)

    def test_set_index_function(addr: int) -> int:
        assert 0b11111 == 0x1F
        assert 0b1110_0000_0000_0000 == 0xE000
        assert 0b10_0000_0000_0000_0000 == 0x20000
        assert 0b1000_0000_0000_0000_0000 == 0x80000

        # Lower xor value is bits 7-11
        lower_xor = (addr >> line_size_log2) & 0b11111

        # Upper xor value is bits 13, 14, 15, 17, and 19
        upper_xor = (addr & 0b1110_0000_0000_0000) >> 13
        # Bit 17
        upper_xor |= (addr & 0b10_0000_0000_0000_0000) >> 14
        # Bit 19
        upper_xor |= (addr & 0b1000_0000_0000_0000_0000) >> 15

        set_index = lower_xor ^ upper_xor
        return set_index

    # build table
    data = []
    for line in range((2**num_bits) // line_size):
        addr = line * line_size
        # print("n", bits(n, num_bits), "set", set)
        data.append([addr, test_set_index_function(addr)])

    data = np.unique(np.array(data), axis=0)
    df = pd.DataFrame(np.array(data), columns=["n", "set"])
    print(df)
    print(compute_set_probability(df))

    # expect:
    # set index 0: b[7] XOR b[13]
    sols = solve_mapping_table_xor(df, num_bits=num_bits)
    # print(sol)


if __name__ == "__main__":
    main()
