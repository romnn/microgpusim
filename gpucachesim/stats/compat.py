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
        # per_config_metric_df = []

        # for (bench_name, bench) in selected_benchmarks_copy.items():
        for bench_name, selected_bench_inputs in selected_benchmarks:
            bench = benchmarks[bench_name]
            # supported_simulators = [
            #     sim_name for sim_name in selected_simulators if bench.enabled(sim_name)  # and inp.enabled(sim_name)
            # ]
            for inp_args, inp_abbr in selected_bench_inputs:

                inp_matches = [i for i in bench.inputs if i.args.strip() == inp_args.strip()]
                assert len(inp_matches) == 1
                inp = inp_matches[0]
                print(config_name, bench_name, inp)

                #     assert inp is not None, f"input {inp_args} does not exist"
                #
                plot_data = PlotData(benchmark=bench, config=config, inp=inp)
                # print(plot_data.m2s_df)

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
                #     per_config_metric_df.append(metric_df)
                all_metric_df.append(metric_df)

        # per_config_metric_df = pd.concat(per_config_metric_df)
        # break
        # continue

        # fig = metrics_plot_func(
        #     data=per_config_metric_df,
        #     config=config,
        #     metric_cls=metric_cls,
        #     title=f"{metric_cls.name} Correlation ({config.name})",
        # )
        # filename = ["scatter", metric_cls.name, config.key]
        # filename = Path("./figs") / gpusims.utils.slugify("_".join(filename))
        # filename = filename.with_suffix(".pdf")
        # fig.write_image(filename, **PDF_OPTS)
        # print("wrote", filename)

    all_metric_df = pd.concat(all_metric_df)
    return all_metric_df


def dedup_ordered(l: list[typing.Any]) -> list[typing.Any]:
    # out = []
    # seen = set()
    # for ll in l:
    #     if ll in seen:
    #         continue
    #     seen.add(ll)
    #     out.append(ll)
    # return out
    return list(dict.fromkeys(l))


def correl_err_table(aggregated, simulators, configs, selected_simulators, selected_configs):

    # selected_simulators = [simulators[s] for s in aggregated.index.get_level_values("Simulator")]
    selected_simulators = [simulators[s] for s in selected_simulators]
    selected_simulators = [s for s in selected_simulators if s.ID != gpusims.NATIVE]
    selected_simulators = dedup_ordered(selected_simulators)

    # selected_configs = [configs[c] for c in aggregated.index.get_level_values("Config")]
    selected_configs = [configs[c] for c in selected_configs]
    # selected_configs = set(dict(test="test"))
    # selected_configs = set(selected_configs)
    # selected_configs = dict.fromkeys(selected_configs)
    selected_configs = dedup_ordered(selected_configs)

    pprint(selected_simulators)
    pprint(selected_configs)

    table = ""

    header_simulators = [""]
    header_configs = [""]
    for sim in selected_simulators:
        header_simulators.append(r"\multicolumn{2}{c|}{" + SIM_NAME_TEX[sim.ID] + "}")
        header_configs += [r"\scriptsize \centering\arraybackslash {}".format([c.name for c in selected_configs])]

    table += "\n & ".join(header_simulators) + r" \\" + "\n"
    table += " % "
    table += "\n & ".join(header_configs) + r" \\ \hline" + "\n"
    table += " %\n"

    # config=conf,
    # sim=sim,
    # corr="{:.3f}".format(correl_co),
    # err=(r"{:.1f}\%".format(rel_err) if rel_err < 100.0 else r"{:.0f}\%".format(rel_err)),
    # nrmse=(r"{:.1f}\%".format(nrmse * 100) if nrmse * 100 < 100.0 else r"{:.0f}\%".format(nrmse * 100)),

    for metric_key, metric_name, is_percent in [
        ("corr", "Corr.", False),
        ("mape", "MAPE", True),
        ("rmspe", "RMSPE", True),
    ]:
        line = [metric_name]
        for sim in selected_simulators:
            for config in selected_configs:
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

    # for metric_key, metric_name in [
    #     ("corr", "Corr."),
    #     ("err", "Rel. Err"),
    #     ("nrmse", "NRMSE"),
    # ]:
    #     line = [metric_name]
    #     # for si, sim in enumerate(selected_simulators):
    #     # for ci, conf in enumerate(selected_configs):
    #     # matches = [e for e in err_data if e["config"] == conf and e["sim"] == sim]
    #     # value = ""
    #     # if len(matches) == 1:
    #     #     value = matches[0][metric_key]
    #     # line.append(value)
    #     table += " & ".join(line) + r" \\" + "\n"
    #     table += " %\n"

    return table


def result_table():
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

    # define ordering that makes sense
    # a4000 is so close to the rtx3070 we exclude it?
    # selected_configs = ["sm6_gtx1080", "sm86_a4000", "sm86_rtx3070"]
    selected_configs = ["sm6_gtx1080", "sm86_rtx3070"]

    # def plot_scatter_subplots(
    #     in_data, metric_cls,
    #     title=None, fontsize=32, font_family="Helvetica", round_to=2
    # ):
    # fig = make_subplots(rows=1, cols=2,
    #                     subplot_titles=["GTX 1080", "RTX 3070"],
    #                     horizontal_spacing=0.15)

    cycles_df_csv_path = RESULTS_DIR / "compat_cycles.csv"
    if False:
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

    # simulator, config, benchmark, value
    # print(cycles_df)

    # pivoted = cycles_df.pivot(index=["Config", "Benchmark"], columns="Simulator", values="Value")
    # pivoted.columns = pivoted.columns.values
    # pivoted = pivoted.reset_index()

    # cycles_df["bench_preview"] = cycles_df["Benchmark"].str[:10]

    unique_benchmarks = sorted(list(cycles_df["Benchmark"].unique()))
    cycles_df["Benchmark"] = cycles_df["Benchmark"].apply(lambda b: unique_benchmarks.index(b))
    cycles_df = cycles_df.set_index(["Config", "Simulator", "Benchmark"])
    # print(cycles_df)
    # print(cycles_df.columns)

    native_values = cycles_df.loc[pd.IndexSlice[:, "native", :], :]
    # print(native_values.columns)
    native_values = native_values.rename(columns={"Value": "native"})
    # native_values.columns = ["native"]
    # joined = cycles_df.join(native_values)
    joined = cycles_df.reset_index().merge(
        # joined = cycles_df.merge(
        # joined = cycles_df.merge(
        # native_values,
        native_values.reset_index(),
        # left_on=["Simulator", "Config", "Benchmark"],
        # left_on=["Config", "Benchmark"],
        # right_on=["Config", "Benchmark"],
        on=["Config", "Benchmark"],
        how="left",
        suffixes=("", "_drop"),
        sort=False,
        # suffixes=("", "_remove"),
    )
    print(joined)
    joined = joined.drop(columns=[col for col in joined if col.endswith("_drop")])
    joined = joined.set_index(["Config", "Simulator", "Benchmark"])

    preview_cols = ["Config", "Simulator", "bench_preview", "Value", "native"]
    # print(joined.reset_index()[preview_cols])

    import gpucachesim.stats.metrics as metric_funcs

    # grouped = joined.reset_index().groupby(["Config"])
    # grouped = joined.groupby(["Config", "Simulator", "Benchmark"])
    grouped = joined.groupby(["Config", "Simulator"], dropna=False, sort=False)
    # grouped.apply(lambda x: print(x.values))

    aggregated = grouped.median()  # aggregate("median")

    def compute_mape(df):
        true_values = df["native"].values
        values = df["Value"].values
        return metric_funcs.mape(true_values=true_values, values=values)

    # aggregated["mape"] = grouped.apply(compute_mape)
    aggregated["mape"] = grouped.apply(lambda df: metric_funcs.mape(true_values=df["native"], values=df["Value"]))
    aggregated["rmspe"] = grouped.apply(lambda df: metric_funcs.rmspe(true_values=df["native"], values=df["Value"]))
    aggregated["corr"] = grouped.apply(
        lambda df: metric_funcs.correlation(true_values=df["native"], values=df["Value"])
    )
    print(aggregated)

    table = correl_err_table(
        aggregated=aggregated,
        simulators=simulators,
        configs=configs,
        selected_simulators=selected_simulators,
        selected_configs=selected_configs,
    )
    print(table)

    return

    def compute_err(df):
        true_values = df["native"].values
        values = df["Value"].values
        corr = metric_funcs.correlation(true_values=true_values, values=values)
        mape = metric_funcs.mape(true_values=true_values, values=values)
        rmspe = metric_funcs.rmspe(true_values=true_values, values=values)
        df["corr"] = corr
        df["mape"] = mape
        df["rmspe"] = rmspe
        # return pd.DataFrame.from_records([("mape", mape), ("rmspe", rmspe)])
        # return pd.DataFrame(dict(uwe=[values.mean()], mape=[mape], rmspe=[rmspe]))
        # return df
        # print(df)
        # print(df.index)
        # print(df.columns)
        # print(df.index)
        # return df.reset_index()
        # print(df.reset_index(drop=True).reset_index(drop=True).T)
        # return df.reset_index(drop=True).reset_index(drop=True)
        # return df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        # print(df.index)
        # df = df.droplevel("Simulator")
        # df = df.droplevel("Benchmark")
        # df = df.droplevel("Config")
        # df.drop(column="Simulator")
        # return df.set_index(drop=True)
        return df
        # return df.drop(columns="Simulator")
        # return df

    # joined = grouped.aggregate({"Value": [compute_err]})
    # joined["Value"] = grouped.apply(compute_err)
    joined = grouped.apply(compute_err)
    joined = joined.droplevel(-1)
    print(joined)
    print(joined.index)
    print(joined.columns)
    # print(joined)
    # print(joined.reset_index()[preview_cols])
    # print(joined[preview_cols])
    # print(joined[["mape", "rmspe"]])
    # print(joined[["mape", "rmspe"]])
    return

    for simulator in selected_simulators:
        true_values = pivoted.loc[:, "native"]
        values = pivoted.loc[:, simulator]
        # true_values = true_values.to_numpy().reshape(-1, 1)
        # values = values.to_numpy().reshape(-1, 1)
        # print(true_values)
        # print(values)

        def compute_mape(a):
            print(a)
            # print(a.values)
            # print(a["native"])
            # print(a[simulator])
            # print(a.index)
            # print(a.values)
            pass

        # mape = pivoted.loc[:, ["native", simulator]].apply(compute_mape)
        # print(a.values)
        # mape = pivoted[pivoted.columns + pivoted.index.columns].apply(compute_mape, axis=1)
        mape = pivoted.apply(compute_mape, axis=1)

        # corr = metric_funcs.correlation(true_values=true_values, values=values)
        # print("corr", corr)

        # mape = metric_funcs.mape(true_values=true_values, values=values)
        # print("mape", mape)

        # rmspe = metric_funcs.rmspe(true_values=true_values, values=values)
        # print("rmspe", rmspe)

        # pivoted.loc[:, simulator + "_corr"] = corr
        pivoted.loc[:, simulator + "_mape"] = mape
        # pivoted.loc[:, simulator + "_rmspe"] = rmspe

    # print(pivoted)
    return
    data = in_data.set_index(["Simulator", "Config"])
    data = data.sort_values(by=["Config", "Benchmark"])
    data = data.sort_index()
    simulators = data.index.get_level_values("Simulator").unique().tolist()
    # configs = data.index.get_level_values("Config").unique().tolist()
    plot_configs = ["sm6_gtx1080", "sm86_rtx3070"]

    all_data = []
    annotations = []
    table_data = []

    for ci, conf in enumerate(plot_configs):
        print(conf)
        vsi = 0
        for si, sim in enumerate(simulators):
            if sim == gpusims.NATIVE:
                continue
            vsi += 1

            sim_values = data.loc[data.index == (sim, conf)]  # .reset_index()
            sim_values = sim_values.rename(columns={"Value": "SimValue"})

            hw_values = data.loc[data.index == (gpusims.NATIVE, conf)]  # .reset_index()
            hw_values = hw_values.rename(columns={"Value": "HwValue"})

            # print(sim_values)
            # print(hw_values)
            values = sim_values.merge(hw_values, on="Benchmark")

            hw_values = values["HwValue"].to_numpy()
            sim_values = values["SimValue"].to_numpy()

            # print(sim, "hw", hw_values)
            # print(sim, "sim", sim_values)

            all_data.append(values)

            if sim_values.sum() > 0:
                correl_co = np.corrcoef(hw_values, sim_values)[0][1]
            else:
                correl_co = 0
            errs = sim_values - hw_values

            rel_errs = np.absolute(errs) / (hw_values + 0.0000001)
            assert rel_errs.shape == errs.shape

            assert len(errs) > 0
            mean_total_err = np.absolute(rel_errs).sum() / len(errs)

            assert hw_values.sum() > 0
            mean_total_agg_err = np.absolute(errs).sum() / hw_values.sum()
            mae = np.absolute(errs).sum() / len(errs)

            assert len(hw_values) > 0
            mse = np.power(errs, 2).sum() / len(hw_values)
            nrmse = np.power(errs, 2).sum() / np.power(hw_values, 2).sum()

            fig.add_trace(
                go.Scatter(
                    x=values["HwValue"],
                    y=values["SimValue"],
                    hovertext=values["Benchmark"],
                    mode="markers",
                    marker=dict(
                        size=16,
                        color="rgba(%d, %d, %d, %f)" % (*hex_to_rgb(SIM_COLOR[sim]), 0.7),
                        symbol="x",
                    ),
                    name=SIM_NAME[sim],
                    showlegend=ci == 0,
                    # name="{}<br>Corr={:.3f}, Err={:.1f}%, NRMSE={:.1f}%<br>".format(
                    #     SIM_NAME[sim], correl_co, mean_total_err*100, nrmse*100),
                ),
                row=1,
                col=ci + 1,
            )

            rel_err = mean_total_err * 100
            table_data.append(
                dict(
                    config=conf,
                    sim=sim,
                    corr="{:.3f}".format(correl_co),
                    err=(r"{:.1f}\%".format(rel_err) if rel_err < 100.0 else r"{:.0f}\%".format(rel_err)),
                    nrmse=(r"{:.1f}\%".format(nrmse * 100) if nrmse * 100 < 100.0 else r"{:.0f}\%".format(nrmse * 100)),
                )
            )

    table = correl_err_table(
        simulators=simulators,
        configs=configs,
        selected_simulators=selected_simulators,
        selected_configs=selected_configs,
    )
    print(table)


def slowdowns():
    pass
