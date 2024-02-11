import pandas as pd
import copy
from wasabi import color
import gpucachesim.utils as utils
import gpucachesim.benchmarks as benchmarks

from gpucachesim import REPO_ROOT_DIR
from gpucachesim.benchmarks import (
    Target,
    Benchmarks,
)


def load_stats(bench_name, profiler="nvprof", path=None) -> pd.DataFrame:
    stats = []
    if bench_name is not None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
            profiler, bench_name
        )
        print("loading {}".format(stats_file))
        df = pd.read_csv(stats_file, header=0)
        if len(df) < 1:
            print(color("WARNING: {} is empty!".format(stats_file), fg="red"))
        else:
            stats.append(df)
    else:
        b = Benchmarks(path)
        benches = utils.flatten(list(b.benchmarks[Target.Profile.value].values()))
        bench_names = set([b["name"] for b in benches])
        for name in bench_names:
            stats_file = REPO_ROOT_DIR / "results/combined.stats.{}.{}.csv".format(
                profiler, name
            )
            print("loading {}".format(stats_file))
            df = pd.read_csv(stats_file, header=0)
            if len(df) < 1:
                print(color("WARNING: {} is empty!".format(stats_file), fg="red"))
            else:
                stats.append(df)

    stats_df = pd.concat(stats, ignore_index=False)
    stats_df = stats_df.sort_values(["benchmark", "target"])
    if bench_name is not None:
        if isinstance(bench_name, str):
            bench_names = [bench_name]
        elif isinstance(bench_name, list):
            bench_names = bench_name
        else:
            raise ValueError
        stats_df = stats_df[stats_df["benchmark"].isin(bench_names)]

    # special_dtypes = {
    #     # **{col: "float64" for col in stats_df.columns},
    #     # **{col: "object" for col in benchmarks.NON_NUMERIC_COLS.keys()},
    #     "target": "str",
    #     "benchmark": "str",
    #     "Host Name": "str",
    #     "Process Name": "str",
    #     "device": "str",
    #     "context_id": "float",
    #     "is_release_build": "bool",
    #     "kernel_function_signature": "str",
    #     "kernel_name": "str",
    #     "kernel_name_mangled": "str",
    #     "input_id": "float",
    #     # "input_memory_only": "first",
    #     # "input_mode": "first",
    #     # makes no sense to aggregate
    #     "cores_per_cluster": "float",
    #     "num_clusters": "float",
    #     "total_cores": "float",
    #     "input_memory_only": "bool",
    #     "input_num_clusters": "float",
    #     "input_cores_per_cluster": "float",
    #     "input_mode": "str",
    #     "input_threads": "float",
    #     "input_run_ahead": "float",
    # }
    # missing_dtypes = set(benchmarks.NON_NUMERIC_COLS.keys()) - set(special_dtypes.keys())
    # assert len(missing_dtypes) == 0, "missing dtypes for {}".format(missing_dtypes)

    dtypes = {
        **{col: "float64" for col in stats_df.columns},
        **benchmarks.SPECIAL_DTYPES,
    }
    # raise ValueError("test")
    dtypes = {col: dtype for col, dtype in dtypes.items() if col in stats_df}
    stats_df = stats_df.astype(dtypes)

    simulation_targets = [
        Target.Simulate.value,
        Target.AccelsimSimulate.value,
        Target.PlaygroundSimulate.value,
    ]
    simulation_targets_df = stats_df[stats_df["target"].isin(simulation_targets)]
    if not (simulation_targets_df["is_release_build"] == True).all():
        print(color("WARNING: non release results:", fg="red"))
        non_release_results = simulation_targets_df[
            simulation_targets_df["is_release_build"] == True
        ]
        grouped = non_release_results.groupby(["benchmark", "target"])
        print(grouped["input_id"].count())
        print("====")

    non_float_cols = set(
        [
            col
            for col, dtype in benchmarks.SPECIAL_DTYPES.items()
            if dtype not in ["float", "float64", "int", "int64"]
        ]
    )
    nan_dtype = pd.NA
    fill = {
        **{col: 0.0 for col in stats_df.columns},
        **{col: nan_dtype for col in non_float_cols},
        **{
            "kernel_name_mangled": nan_dtype,
            "kernel_name": nan_dtype,
            "device": nan_dtype,
            # test this out
            "kernel_launch_id": nan_dtype,
            "run": nan_dtype,
        },
        **{c: nan_dtype for c in benchmarks.ALL_BENCHMARK_INPUT_COLS},
        **{c: nan_dtype for c in benchmarks.SIMULATE_INPUT_COLS},
        **{
            "input_memory_only": False,
            "input_num_clusters": 28,
            "input_cores_per_cluster": 1,
        },
    }
    assert pd.isnull(fill["kernel_launch_id"])
    assert pd.isnull(fill["kernel_name"])
    # fill = {
    #     col: dtype for col, dtype in fill.items()
    #     if col not in benchmarks.CATEGORICAL_COLS
    # }

    stats_df = stats_df.fillna(fill).infer_objects(copy=False)
    assert stats_df["run"].isna().sum() == 0

    def add_no_kernel_exec_time(df):
        # print(df[benchmarks.PREVIEW_COLS][:4].T)
        try:
            before = copy.deepcopy(df.dtypes)
            if df["target"].iloc[0] != Target.Simulate.value:
                return df

            assert (
                len(df) >= 2
            ), "expected at least two rows: a no kernel row and at least one kernel for the config"
            # print("df")
            # print(df)
            valid_kernels = ~df["kernel_name"].isna()
            # print("valid_kernels")
            # print(valid_kernels)
            no_kernel = df[~valid_kernels]
            # print("no kernel")
            # print(no_kernel)
            assert len(no_kernel) == 1
            num_valid_kernels = valid_kernels.sum()
            assert num_valid_kernels >= 1
            delta = float(no_kernel["exec_time_sec"].iloc[0]) / num_valid_kernels
            df.loc[valid_kernels, "exec_time_sec"] += delta
            assert (df.dtypes == before).all()
            return df
        except Exception as e:
            print(e)
            return str(e)

    group_cols = list(
        benchmarks.BENCH_TARGET_INDEX_COLS
        + list(benchmarks.ALL_BENCHMARK_INPUT_COLS)
        + benchmarks.SIMULATE_INPUT_COLS
        + ["run"]
    )
    print(len(stats_df))
    group_cols = [col for col in group_cols if col in stats_df]
    # pprint(group_cols)
    # pprint(stats_df[group_cols].dtypes)
    # stats_df = stats_df.fillna({'target': "", "benchmark": "", "input_mode": ""})
    grouped = stats_df.groupby(group_cols, dropna=False)
    # grouped = grouped[stats_df.columns].fillna({'target': "", "benchmark": "", "input_mode": ""})
    # print(grouped.isna())
    # raise ValueError("grouped")
    stats_df = grouped[stats_df.columns].apply(add_no_kernel_exec_time)
    stats_df = stats_df.reset_index(drop=True)
    # raise ValueError("its over")

    assert stats_df["run"].isna().sum() == 0
    # assert stats_df["kernel_launch_id"].isna().sum() == 0
    assert stats_df["num_clusters"].isna().sum() == 0
    return stats_df
