import click
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import gpucachesim.plot as plot
import gpucachesim.utils as utils
from gpucachesim.benchmarks import REPO_ROOT_DIR


@click.command()
# @click.option("--path", help="Path to materialized benchmark config")
# @click.option("--bench", "bench_name", help="Benchmark name")
# @click.option("--nvprof", "nvprof", type=bool, is_flag=True, help="use nvprof")
def main():
    stats_file = REPO_ROOT_DIR / "results/trace-metrics.csv"
    df = pd.read_csv(stats_file, header=0)
    df["MB"] = df["num_bytes"].astype(float) / float(1024**2)
    df["KI"] = df["num_instructions"].astype(float) / 1000.0
    df["KI/sec"] = df["KI"].astype(float) / df["deserialization_time_sec"].astype(float)
    df["MB/sec"] = df["MB"].astype(float) / df["deserialization_time_sec"].astype(float)
    df["MB/KI"] = df["MB"].astype(float) / df["KI"].astype(float)

    print(
        df[
            [
                "format",
                "num_instructions",
                "deserialization_time_sec",
                "KI",
                "MB",
                "KI/sec",
                "MB/sec",
                "MB/KI",
            ]
        ]
    )
    print(df.groupby("format")[["KI/sec", "MB/sec", "MB/KI"]].mean())

    table = r" & Size & \multicolumn{2}{c}{Deserialization speed} \\"
    table += "\n"
    table += r" & MiB/KI & MiB/sec & KI/sec \\ \hline"
    table += "\n"
    for row_idx, (format, per_format) in enumerate(df.groupby("format")):
        # print(format)
        # print(per_format["MB/KI"].values)
        ki_per_sec = per_format["KI/sec"].mean()
        mb_per_sec = per_format["MB/sec"].mean()
        mb_per_ki = per_format["MB/KI"].mean()
        if row_idx % 2 == 0:
            table += r"\rowcolor{gray!10}"
        table += (
            str(format)
            + " & "
            + "{:>4.2f}".format(mb_per_ki)
            + " & "
            + "{:>5.2f}".format(mb_per_sec)
            + " & "
            + "{:>6.2f}".format(ki_per_sec)
            + r" \\"
        )
        table += "\n"

    print(table)


if __name__ == "__main__":
    main()
