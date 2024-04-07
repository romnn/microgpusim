import sys
import os
import re
import copy
import shlex
import typing
import subprocess as sp
from pathlib import Path
from pprint import pprint
from io import BytesIO


DIR = Path(__file__).parent
PARBOIL = DIR / "benchmarks"
RODINIA = DIR / "rodinia_3.1"
CUDA_12_4_SDK = DIR / "cuda-samples-12.4"

PARBOIL_BENCHMARKS = sorted(list(PARBOIL.rglob("*/src/cuda/Makefile")))
RODINIA_BENCHMARKS = sorted(list(RODINIA.rglob("cuda/*/Makefile")))
CUDA_12_4_SDK_BENCHMARKS = sorted(list(CUDA_12_4_SDK.rglob("Samples/*/*/Makefile")))
# ALL_BENCHMARKS = PARBOIL_BENCHMARKS + RODINIA_BENCHMARKS + CUDA_12_4_SDK_BENCHMARKS
ALL_BENCHMARKS = CUDA_12_4_SDK_BENCHMARKS
BAD_BENCHMARKS = [
    "histEqualizationNPP",
    "conjugateGradientMultiBlockCG",
    "conjugateGradientMultiDeviceCG",
    "cudaNvSciNvMedia",
    "cudaNvSci",
    "immaTensorCoreGemm",
    "cudaTensorCoreGemm",
    "systemWideAtomics",
    "fp16ScalarProduct",
]

MIN = 60

SMS = [
    # fermi
    20,
    # kepler
    30,
    35,
    37,
    # maxwell
    50,
    52,
    53,
    # pascal
    60,
    61,
    62,
    # volta
    70,
    72,
    # turing
    75,
    # ampere
    80,
    86,
    87,
    # ada
    89,
    # hopper
    90,
]
assert sorted(SMS) == SMS


class ExecError(Exception):
    def __init__(self, msg, cmd, stdout, stderr):
        self.cmd = cmd
        self.stdout = stdout
        self.stderr = stderr
        super().__init__(msg)


class ExecStatusError(ExecError):
    def __init__(self, cmd, status, stdout, stderr):
        self.status = status
        super().__init__(
            "command {} completed with non-zero exit code ({})".format(cmd, status),
            cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )


class ExecTimeoutError(ExecError):
    def __init__(self, cmd, timeout, stdout, stderr):
        self.timeout = timeout
        super().__init__(
            "command {} timed out after {} seconds".format(cmd, timeout),
            cmd=cmd,
            stdout=stdout,
            stderr=stderr,
        )


def run_cmd(
    cmd,
    cwd=None,
    shell=False,
    timeout_sec=None,
    env=None,
    verbose=False,
    stream_output=False,
):
    if not shell and not isinstance(cmd, list):
        cmd = shlex.split(cmd)

    proc = sp.Popen(
        cmd,
        stdout=sp.PIPE,
        stderr=sp.PIPE,
        cwd=cwd,
        env=env,
        shell=shell,
        encoding=None,
    )

    try:
        stdout = BytesIO()
        stderr = BytesIO()
        if stream_output:
            os.set_blocking(proc.stdout.fileno(), False)
            os.set_blocking(proc.stderr.fileno(), False)
            buffer_size = 1024
            while True:
                terminated = proc.poll() is not None

                if proc.stderr.readable():
                    stderr_buffer = proc.stderr.read(-1 if terminated else buffer_size)
                    if stderr_buffer is not None:
                        sys.stderr.buffer.write(stderr_buffer)
                        stderr.write(stderr_buffer)

                if proc.stdout.readable():
                    stdout_buffer = proc.stdout.read(-1 if terminated else buffer_size)
                    if stdout_buffer is not None:
                        stdout.write(stdout_buffer)
                sys.stderr.buffer.flush()

                if terminated:
                    break

            stdout.seek(0)
            stdout = stdout.read()
            stderr.seek(0)
            stderr = stderr.read()
        else:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
    except sp.TimeoutExpired as timeout:
        proc.kill()
        stdout, stderr = proc.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")

        print("\n{} timed out\n".format(cmd))
        print("\nstdout (last 15 lines):\n")
        print("\n".join(stdout.splitlines()[-15:]))
        print("\nstderr (last 15 lines):\n")
        print("\n".join(stderr.splitlines()[-15:]))

        sys.stdout.flush()

        raise ExecTimeoutError(
            cmd=cmd,
            timeout=timeout.timeout,
            stdout=stdout,
            stderr=stderr,
        )

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if proc.returncode != 0:
        print("\nstdout (last 15 lines):\n")
        print("\n".join(stdout.splitlines()[-15:]))
        print("\nstderr (last 15 lines):\n")
        print("\n".join(stderr.splitlines()[-15:]))
        sys.stdout.flush()
        raise ExecStatusError(cmd=cmd, status=proc.returncode, stdout=stdout, stderr=stderr)

    # command succeeded
    return proc.returncode, stdout, stderr


def clean(makefile: Path):
    print("cleaning {}".format(makefile.parent))
    ret_code, _, _ = run_cmd(["make", "-B", "-C", str(makefile.parent.absolute()), "clean"])
    assert ret_code == 0


def is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def sanitize_filename(value: str, allow_unicode=False) -> str:
    import unicodedata

    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


def build_benchmarks(output_dir=None):
    _, nvcc_version_stdout, _ = run_cmd(["nvcc", "--version"])
    cuda_major, cuda_minor, cuda_rest = extract_cuda_version(nvcc_version_stdout)
    cuda_rest = sanitize_filename(cuda_rest)

    need_clean = []

    for sm in SMS:
        for makefile in need_clean:
            clean(makefile)
        need_clean = []
        valid = 0
        for makefile in ALL_BENCHMARKS:
            bench_dir = makefile.parent.relative_to(DIR)
            name = bench_dir.name
            print("[SM {:<2}] building {}".format(sm, bench_dir))

            need_clean.append(makefile)
            try:
                cmd = ["make", "-B", "-C", str(makefile.parent.absolute()), "build"]
                print(" ".join(cmd))
                ret_code, stdout, stderr = run_cmd(
                    cmd,
                    env={**os.environ, **{"SMS": str(sm)}},
                )
            except ExecStatusError as e:
                print("stdout:")
                print(e.stdout)
                print("stderr:")
                print(e.stderr)
                if "Unsupported gpu architecture" in e.stderr:
                    break
                elif re.search("only allowed for architecture compute_\d+ or later", e.stderr):
                    continue
                elif re.search(r"are only supported for sm_\d+ and up", e.stderr):
                    continue
                elif name in BAD_BENCHMARKS:
                    continue
                else:
                    raise e

            executables = [f for f in makefile.parent.iterdir() if f.suffix == "" and is_executable(f)]
            pprint(executables)
            if len(executables) == 1:
                valid += 1
                assert executables[0].name == makefile.parent.name

            elif re.search("is not supported on Linux .* waiving sample", stdout + stderr):
                print("skipping {}".format(bench_dir))
                continue

            elif re.search(r"Waiving build\. GLIBC > [\d\.]+ is not supported", stdout + stderr):
                print("skipping {}".format(bench_dir))
                continue

            elif name in BAD_BENCHMARKS:
                continue

            else:
                raise ValueError("no executable for {}".format(bench_dir))

            if output_dir is not None:
                for executable in executables:
                    # extract the ptx files
                    _, ptx_names, _ = run_cmd(
                        ["cuobjdump", "-lptx", str(executable.absolute())],
                        timeout_sec=1 * MIN,
                    )
                    ptx_names = extract_ptx_file_names(ptx_names)
                    for i, (_, ptx_name) in enumerate(ptx_names):
                        stem = [
                            "cuda_{}_{}_{}".format(cuda_major, cuda_minor, cuda_rest),
                            "sm{}".format(sm),
                            ptx_name,
                        ]

                        dest = Path(output_dir) / "_".join(stem)
                        dest = dest.with_suffix(".ptx")
                        print(
                            "{} [{:>2}/{:<2}]: extract ptx {} to {}".format(
                                executable.name, i+1, len(ptx_names), ptx_name, dest
                            )
                        )

                        run_cmd(
                            [
                                "cuobjdump",
                                "-xptx",
                                ptx_name,
                                str(executable.absolute()),
                            ],
                            cwd=Path(output_dir),
                            timeout_sec=1 * MIN,
                        )

                        # rename
                        (Path(output_dir) / ptx_name).rename(dest)
        if valid > 0:
            break
        print("[SM {:<2}] done".format(sm))
    print("done")


def extract_cuda_version(s: str) -> typing.Tuple[int, int, str]:
    match = re.search(r"Build cuda_(\d+).(\d+).(\S*)", s, re.MULTILINE)
    major, minor, rest = match.groups()
    return major, minor, rest


def extract_ptx_file_names(s: str) -> list[typing.Tuple[int, str]]:
    matches = re.findall(r"PTX file\s+(\d+):\s+([^\n]+)", s)
    # return [m.groups() for m in matches]
    return matches


if __name__ == "__main__":
    print(len(PARBOIL_BENCHMARKS))
    print(len(RODINIA_BENCHMARKS))
    print(len(CUDA_12_4_SDK_BENCHMARKS))

    test = """
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Wed_Nov_22_10:17:15_PST_2023
Cuda compilation tools, release 12.3, V12.3.107
Build cuda_12.3.r12.3/compiler.33567101_0
"""

    test2 = """
PTX file    1: asyncAPI.1.sm_50.ptx
PTX file    2: asyncAPI.2.sm_50.ptx
"""

    # print(extract_ptx_file_names(test2))

    # sys.exit(0)
    #     res = re.search(
    #         r"Waiving build\. GLIBC > [\d\.]+ is not supported",
    #         """>> Waiving build. GLIBC > 2.33 is not supported<<<
    # [@] /usr/local/cuda/bin/nvcc -ccbin g++""",
    #     )
    #     print(res)
    try:
        output_dir = sys.argv[1]
    except IndexError:
        output_dir = None

    print("output dir", output_dir)
    if output_dir is not None:
        assert Path(output_dir).is_dir()
    # sys.exit(0)

    for makefile in ALL_BENCHMARKS:
        clean(makefile)

    build_benchmarks(output_dir=output_dir)
