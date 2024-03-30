import typing
import pandas as pd
import numpy as np
from copy import copy
from pprint import pprint
from pathlib import Path

from gpucachesim import REPO_ROOT_DIR

RESULTS_DIR = REPO_ROOT_DIR / "results"
assert RESULTS_DIR.is_dir()

root_dir = Path("/home/roman/dev/simulators")
assert root_dir.is_dir()
benchmark_dir = root_dir / "benchmarks"
run_dir = root_dir / "run"
assert benchmark_dir.is_dir()
assert run_dir.is_dir()

import sys

sys.path.insert(0, str(root_dir))
import gpusims
import gpusims.plot.metrics as metric
import gpucachesim.stats.metrics as metric_funcs
from gpusims.plot.data import PlotData
from gpusims.config import Config, parse_configs
from gpusims.bench import parse_benchmarks

pd.set_option("display.max_rows", 700)
pd.set_option("display.max_columns", 700)
np.seterr(all="raise")

SIM_NAME = {
    gpusims.TEJAS: "GPUTejas",
    gpusims.ACCELSIM_PTX: "AccelSim PTX",
    gpusims.ACCELSIM_SASS: "AccelSim SASS",
    gpusims.NATIVE: "Hardware",
    gpusims.MULTI2SIM: "Multi2Sim",
    gpusims.MACSIM: "MacSim",
}

SIM_NAME_TEX = {
    gpusims.TEJAS: r"\textsc{GpuTejas}",
    gpusims.ACCELSIM_PTX: r"\textsc{AccelSim PTX}",
    gpusims.ACCELSIM_SASS: r"\textsc{AccelSim SASS}",
    gpusims.NATIVE: r"Hardware",
    gpusims.MULTI2SIM: r"\textsc{Multi2Sim}",
    gpusims.MACSIM: r"\textsc{MacSim}",
}


def build_metric_df(
    metric_cls,
    simulators,
    benchmarks,
    configs,
    selected_benchmarks,
    selected_configs,
    selected_simulators,
):
    all_metric_df = []
    for config_name in selected_configs:
        config = configs[config_name]

        for bench_name, selected_bench_inputs in selected_benchmarks:
            bench = benchmarks[bench_name]
            for inp_args, _ in selected_bench_inputs:

                inp_matches = [i for i in bench.inputs if i.args.strip() == inp_args.strip()]
                assert len(inp_matches) == 1
                inp = inp_matches[0]
                print(config_name, bench_name, inp)

                plot_data = PlotData(benchmark=bench, config=config, inp=inp)

                for sim_name in selected_simulators:
                    sim = simulators[sim_name]

                    if not bench.enabled(sim_name):
                        continue
                    if not inp.enabled(sim_name):
                        continue

                    # print(sim_name, config_name, bench_name)
                    bench_config = sim(
                        run_dir=run_dir / sim_name.lower(),
                        benchmark=bench,
                        config=config,
                    )
                    if not bench_config.input_path(inp).is_dir():
                        raise ValueError(f"WARN: {bench_config.input_path(inp)} does not exist")

                    plot_data[sim_name] = bench_config.load_dataframe(inp)

                metric = metric_cls(plot_data)
                metric_df = metric.compute()
                metric_df["Benchmark"] = f"{bench.name}<br>{inp.args}"
                metric_df["Config"] = config_name
                all_metric_df.append(metric_df)

    all_metric_df = pd.concat(all_metric_df)
    return all_metric_df


def dedup_ordered(l: list[typing.Any]) -> list[typing.Any]:
    return list(dict.fromkeys(l))


# def build_result_table(aggregated, simulators, configs, selected_simulators, selected_configs):
def build_result_table(aggregated, configs, simulators):

    # selected_simulators = [simulators[s] for s in aggregated.index.get_level_values("Simulator")]

    table = ""

    header_simulators = [""]
    header_configs = [""]
    for sim in simulators:
        header_simulators.append(r"\multicolumn{2}{c|}{" + SIM_NAME_TEX[sim.ID] + "}")
        header_configs += [r"\scriptsize \centering\arraybackslash {}".format([c.name for c in configs])]

    table += "\n & ".join(header_simulators) + r" \\" + "\n"
    table += " % "
    table += "\n & ".join(header_configs) + r" \\ \hline" + "\n"
    table += " %\n"

    for metric_key, metric_name, is_percent in [
        ("corr", "Corr.", False),
        ("mape", "MAPE", True),
        ("rmspe", "RMSPE", True),
    ]:
        line = [metric_name]
        for sim in simulators:
            for config in configs:
                value = aggregated.loc[(config.key, sim.ID), metric_key]
                if is_percent:
                    value *= 100
                    value = r"${value:.{precision}f}\%$".format(value=value, precision=1 if abs(value) < 100.0 else 0)
                else:
                    value = "${:.3f}$".format(value)
                line.append(value)
                pass
        table += " & ".join(line) + r" \\" + "\n"
        table += " %\n"

    return table


def result_table(force=False, verbose=False):
    simulators = copy(gpusims.SIMULATORS)
    configs = parse_configs(benchmark_dir / "configs" / "configs.yml")
    benchmarks = parse_benchmarks(benchmark_dir / "benchmarks.yml")

    pprint(simulators)
    pprint(configs)
    pprint(benchmarks)

    # define ordering that makes sense (e.g. hw and accel close)
    selected_simulators = [
        gpusims.TEJAS,
        gpusims.MACSIM,
        gpusims.MULTI2SIM,
        gpusims.ACCELSIM_PTX,
        gpusims.ACCELSIM_SASS,
        gpusims.NATIVE,
    ]

    # define ordering that makes sense
    # a4000 is so close to the rtx3070 we exclude it?
    # selected_configs = ["sm6_gtx1080", "sm86_a4000", "sm86_rtx3070"]
    selected_configs = ["sm6_gtx1080", "sm86_rtx3070"]

    selected_simulators = [simulators[s] for s in selected_simulators]
    selected_simulators = [s for s in selected_simulators if s.ID != gpusims.NATIVE]
    selected_simulators = dedup_ordered(selected_simulators)

    selected_configs = [configs[c] for c in selected_configs]
    selected_configs = dedup_ordered(selected_configs)

    pprint(selected_simulators)
    pprint(selected_configs)

    # define ordering of inputs that makes sense
    selected_benchmarks = [
        (
            "babelstream",
            [
                ("--arraysize 1024 --numtimes 1", "1024"),
                ("--arraysize 10240 --numtimes 1", "10240"),
                ("--arraysize 102400 --numtimes 1", "102400"),
                # ("--arraysize 1024 --numtimes 2", "1024 (2x)"),
            ],
        ),
        (
            "vectoradd",
            [
                # [inp.args for inp in benchmarks["vectoradd"].inputs]),
                ("1000", "1K"),
                ("1000000", "1M"),
            ],
        ),
        (
            "cuda4-matrixmul",
            [
                # [inp.args for inp in benchmarks["cuda4-matrixmul"].inputs]),
                ("32", "32x32"),
                ("128", "128x128"),
                ("512", "512x512"),
            ],
        ),
        (
            "cuda10-matrixmul",
            [
                ("-wA=32 -hA=32 -wB=32 -hB=32", "32x32 32x32"),
                ("-wA=128 -hA=128 -wB=128 -hB=128", "128x128 128x128"),
                ("-wA=512 -hA=512 -wB=512 -hB=512", "512x512 512x512"),
                # ("-wA=32 -hA=64 -wB=64 -hB=32", "32x64 64x32"),
            ],
        ),
        (
            "cuda6-transpose",
            [
                ("-repeat=1 -dimX=32 -dimY=32", "32x32"),
                ("-repeat=1 -dimX=64 -dimY=64", "64x64"),
                ("-repeat=1 -dimX=128 -dimY=128", "128x128"),
                # ("-repeat=3 -dimX=32 -dimY=32", "32x32 (3x)"),
            ],
        ),
        (
            "cuda10-transpose",
            [
                ("-repeat=1 -dimX=32 -dimY=32", "32x32"),
                ("-repeat=1 -dimX=64 -dimY=64", "64x64"),
                ("-repeat=1 -dimX=128 -dimY=128", "128x128"),
                # ("-repeat=3 -dimX=32 -dimY=32", "32x32"),
            ],
        ),
    ]

    cycles_df_csv_path = RESULTS_DIR / "compat_cycles.csv"
    if force:
        print("generating", cycles_df_csv_path)
        cycles_df = build_metric_df(
            metric_cls=gpusims.plot.metrics.Cycles,
            simulators=simulators,
            benchmarks=benchmarks,
            configs=configs,
            selected_simulators=selected_simulators,
            selected_benchmarks=selected_benchmarks,
            selected_configs=selected_configs,
        )
        cycles_df.to_csv(cycles_df_csv_path, index=False)
    else:
        cycles_df = pd.read_csv(
            cycles_df_csv_path,
            header=0,
        )

    unique_benchmarks = sorted(list(cycles_df["Benchmark"].unique()))
    cycles_df["Benchmark"] = cycles_df["Benchmark"].apply(lambda b: unique_benchmarks.index(b))
    cycles_df = cycles_df.set_index(["Config", "Simulator", "Benchmark"])

    native_values = cycles_df.loc[pd.IndexSlice[:, "native", :], :]
    native_values = native_values.rename(columns={"Value": "native"})
    joined = cycles_df.reset_index().merge(
        native_values.reset_index(),
        on=["Config", "Benchmark"],
        how="left",
        suffixes=("", "_drop"),
        sort=False,
    )
    joined = joined.drop(columns=[col for col in joined if col.endswith("_drop")])
    joined = joined.set_index(["Config", "Simulator", "Benchmark"])

    # preview_cols = ["Config", "Simulator", "bench_preview", "Value", "native"]
    # print(joined.reset_index()[preview_cols])

    grouped = joined.groupby(["Config", "Simulator"], dropna=False, sort=False)

    aggregated = grouped.median()

    aggregated["mape"] = grouped.apply(lambda df: metric_funcs.mape(true_values=df["native"], values=df["Value"]))
    aggregated["rmspe"] = grouped.apply(lambda df: metric_funcs.rmspe(true_values=df["native"], values=df["Value"]))
    aggregated["corr"] = grouped.apply(
        lambda df: metric_funcs.correlation(true_values=df["native"], values=df["Value"])
    )
    print(aggregated)

    table = build_result_table(
        aggregated=aggregated,
        # simulators=simulators,
        # configs=configs,
        simulators=selected_simulators,
        configs=selected_configs,
    )
    print(table)


def slowdowns():
    pass
