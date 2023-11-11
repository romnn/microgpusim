import sys
import os
import re
import copy
import shlex
import subprocess as sp
from timeit import default_timer as timer
from pathlib import Path
from io import BytesIO


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
    save_to=None,
    retries=1,
    dry_run=False,
    verbose=False,
    stream_output=False,
):
    if not shell and not isinstance(cmd, list):
        cmd = shlex.split(cmd)

    err = None
    for attempt in range(retries):
        if verbose:
            print("running {} (attempt {}/{})".format(cmd, attempt + 1, retries))
            if isinstance(cmd, list):
                print("running {} (attempt {}/{})".format(" ".join(cmd), attempt + 1, retries))

        if dry_run:
            return 0, "", "", 0

        if isinstance(cwd, Path):
            cwd = str(cwd.absolute())

        if isinstance(timeout_sec, (int, float)):
            if timeout_sec > 2**20:
                timeout_sec = None

        # the subprocess may take a long time, hence flush all buffers before
        sys.stdout.flush()
        start = timer()
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
                # pass
                # while proc.poll() is None:  # Check the the child process is still running
                #     print("reading now")
                #     data = proc.stdout.read(1)
                #     print("read", data)
                # completed = False
                # i = 0
                # while proc.stdout.readable():  # proc.poll():
                buffer_size = 1024
                while True:
                    terminated = proc.poll() is not None

                    if proc.stderr.readable():
                        stderr_buffer = proc.stderr.read(-1 if terminated else buffer_size)
                        if stderr_buffer is not None:
                            sys.stderr.buffer.write(stderr_buffer)
                            stderr.write(stderr_buffer)

                            # sys.stderr.buffer.write(stderr_buffer)
                            # stderr.write(bytes(stderr_buffer, encoding="utf-8"))

                    if proc.stdout.readable():
                        stdout_buffer = proc.stdout.read(-1 if terminated else buffer_size)
                        if stdout_buffer is not None:
                            # sys.stdout.buffer.write(stdout_buffer)
                            stdout.write(stdout_buffer)

                            # sys.stdout.buffer.write(stdout_buffer)
                            # stdout.write(bytes(stdout_buffer, encoding="utf-8"))
                    sys.stderr.buffer.flush()

                    if terminated:
                        break
                    # pos = stderr.tell()
                    # stderr.seek(0)
                    # lines = stderr.readlines()
                    # stderr.seek(pos)

                    # print(lines)
                    # if len(lines) > 0:
                    #     # if i % 1000 == 0:
                    #     #     print(lines[-1])
                    #     last_line = lines[-1].decode("utf-8")
                    #     if re.match(
                    #         r"completed in \d+\.\d+s",
                    #         last_line,
                    #     ):
                    #         completed = True

                stdout.seek(0)
                stdout = stdout.read()
                stderr.seek(0)
                stderr = stderr.read()
            else:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
        except sp.TimeoutExpired as timeout:
            proc.kill()
            stdout, stderr = proc.communicate()
            # output = ""
            # if isinstance(timeout.output, bytes):
            #     output = timeout.output.decode("utf-8")
            stdout = stdout.decode("utf-8")
            stderr = stderr.decode("utf-8")

            print("\n{} timed out\n".format(cmd))
            print("\nstdout (last 15 lines):\n")
            print("\n".join(stdout.splitlines()[-15:]))
            print("\nstderr (last 15 lines):\n")
            print("\n".join(stderr.splitlines()[-15:]))

            sys.stdout.flush()

            err = ExecTimeoutError(
                cmd=cmd,
                timeout=timeout.timeout,
                stdout=stdout,
                stderr=stderr,
            )
            # try again
            continue

        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        # print(stderr)

        if save_to is not None:
            with open(str((save_to.parent / (save_to.name + ".stdout")).absolute()), "w") as f:
                f.write(stdout)
            with open(str((save_to.parent / (save_to.name + ".stderr")).absolute()), "w") as f:
                f.write(stderr)

        if proc.returncode != 0:
            print("\nstdout (last 15 lines):\n")
            print("\n".join(stdout.splitlines()[-15:]))
            print("\nstderr (last 15 lines):\n")
            print("\n".join(stderr.splitlines()[-15:]))
            sys.stdout.flush()
            err = ExecStatusError(cmd=cmd, status=proc.returncode, stdout=stdout, stderr=stderr)
            # try again
            continue

        # command succeeded
        end = timer()
        duration = end - start
        return proc.returncode, stdout, stderr, duration

    if err is not None:
        raise err
