import click
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

import gpucachesim.stats.stats as stats
import gpucachesim.stats.native as native
import gpucachesim.stats.accelsim as accelsim
import gpucachesim.stats.playground as playground
import gpucachesim.benchmarks as benchmarks
from gpucachesim.stats.human import human_readable
from gpucachesim.benchmarks import Target, Benchmarks, GPUConfig, REPO_ROOT_DIR


# suppress scientific notation by setting float_format
# pd.options.display.float_format = "{:.3f}".format
pd.options.display.float_format = "{:.2f}".format
pd.set_option("display.max_rows", 500)
# pd.set_option("display.max_columns", 500)
# pd.set_option("max_colwidth", 2000)
# pd.set_option("display.expand_frame_repr", False)
np.seterr(all="raise")

DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.config.yml"


@click.group()
# @click.pass_context
def main():
    # ctx.ensure_object(dict)
    pass


SIMULATE_INPUT_COLS = [
    "input_mode",
    "input_threads",
    "input_run_ahead",
    "input_memory_only",
    "input_num_clusters",
    "input_cores_per_cluster",
]

BENCHMARK_INPUT_COLS = {
    "vectorAdd": ["input_dtype", "input_length"],
    "matrixmul": ["input_dtype", "input_rows"],
    "simple_matrixmul": ["input_dtype", "input_m", "input_n", "input_p"],
    "transpose": ["input_dim", "input_variant"],
    "babelstream": ["input_size"],
}

STAT_COLS = [
    "exec_time_sec",
    "cycles",
    "num_blocks",
    "instructions",
    "warp_inst",
    # dram stats
    "dram_reads",
    "dram_writes",
    # l2 stats
    "l2_accesses",
    "l2_reads",
    "l2_writes",
    "l2_read_hit_rate",
    "l2_write_hit_rate",
    "l2_read_miss_rate",
    "l2_write_miss_rate",
    "l2_read_hits",
    "l2_write_hits",
    "l2_read_misses",
    "l2_write_misses",
    "l2_hits",
    "l2_misses",
    # l1 rates
    "l1_hit_rate",
    "l1_miss_rate",
    # l1 accesses
    "l1_reads",
    "l1_writes",
    "l1_hits",
    "l1_misses",
    "l1_accesses",
]

INDEX_COLS = ["target", "benchmark", "input_id"]


def benchmark_results(sim_df: pd.DataFrame, bench_name: str, targets=None) -> pd.DataFrame:
    """View results for a benchmark"""

    selected_df = sim_df.copy()
    selected_df = selected_df[selected_df["benchmark"] == bench_name]
    # print(selected_df)
    # only compare serial gpucachesim
    # selected_df = selected_df[selected_df["input_mode"] != "nondeterministic"]

    for col in SIMULATE_INPUT_COLS:
        if col not in selected_df:
            selected_df[col] = np.nan

    non_gpucachesim = selected_df["input_mode"].isnull()
    print(selected_df[non_gpucachesim]["target"].unique().tolist())

    serial_gpucachesim = selected_df["input_mode"] == "serial"
    compute_gpucachesim = selected_df["input_memory_only"] == False
    gtx1080_gpucachesim = selected_df["input_cores_per_cluster"] == 1
    gtx1080_gpucachesim &= selected_df["input_num_clusters"] == 20
    gold_gpucachesim = serial_gpucachesim & compute_gpucachesim & gtx1080_gpucachesim
    print(
        "gpucachesim gold input ids:",
        sorted(selected_df.loc[gold_gpucachesim, "input_id"].unique().tolist()),
    )

    # only keep gold gpucachesim and other targets
    # selected_df = selected_df[gold_gpucachesim ^ non_gpucachesim]
    # kernels = selected_df[non_gpucachesim][["kernel_name_mangled", "kernel_name"]].drop_duplicates()
    # print(kernels)
    #
    print(selected_df[gold_gpucachesim][["kernel_name_mangled", "kernel_name"]].drop_duplicates())
    # kernels = selected_df[gold_gpucachesim][["kernel_name_mangled", "kernel_name"]].drop_duplicates()
    kernels = selected_df[gold_gpucachesim]["kernel_name"].unique().tolist()
    print(kernels)
    # print(selected_df[gold_gpucachesim][["target", "benchmark", "input_id", "kernel_name_mangled", "cycles"]])
    # print(
    #     selected_df[non_gpucachesim][
    #         ["target", "kernel_name_mangled", "kernel_name", "kernel_launch_id"]
    #     ].drop_duplicates()
    # )

    no_kernel = selected_df["kernel_name"].isna() ^ (selected_df["kernel_name"] == "")
    valid_kernel = selected_df["kernel_name"].isin(kernels)
    selected_df = selected_df[(gold_gpucachesim ^ non_gpucachesim) & (valid_kernel ^ no_kernel)]

    if isinstance(targets, list):
        selected_df = selected_df[selected_df["target"].isin(targets)]

    # assert (selected_df["is_release_build"] == True).all()

    input_cols = BENCHMARK_INPUT_COLS[bench_name]
    # print(selected_df[input_cols].drop_duplicates())

    grouped = selected_df.groupby(INDEX_COLS, dropna=False)
    # print(selected_df[INDEX_COLS + input_cols + ["dram_writes", "l2_accesses"]].head(n=200))
    # print(grouped["dram_writes"].sum())
    averaged = grouped[STAT_COLS + input_cols].mean().reset_index()
    # print(averaged)
    # print(averaged.drop_duplicates())

    per_target = averaged.pivot(index=["benchmark"] + input_cols, columns="target", values=STAT_COLS)
    return per_target


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
# @click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
# @click.option("--input", "input_idx", type=int, help="Input index")
def view(path, bench_name):
    # load the materialized benchmark config
    if bench_name is None:
        stats_file = REPO_ROOT_DIR / "results/combined.stats.csv"
    else:
        stats_file = REPO_ROOT_DIR / f"results/combined.stats.{bench_name}.csv"

    sim_df = pd.read_csv(stats_file, header=0)
    # assert (sim_df["input_mode"] == "serial").sum() > 0

    # print(sim_df)

    per_target = benchmark_results(sim_df, bench_name)
    per_target = per_target[
        [
            "num_blocks",
            "exec_time_sec",
            "cycles",
            "instructions",
            "dram_reads",
            "dram_writes",
            # l2 stats
            "l2_accesses",
            "l2_reads",
            "l2_read_hit_rate",
            "l2_read_hits",
            "l2_writes",
            "l2_write_hits",
            "l2_write_hit_rate",
            # "l2_hits",
            # "l2_misses",
            # l1 stats
            "l1_accesses",
            # "l1_reads",
            # "l1_hits",
            # "l1_misses",
        ]
    ]
    print(per_target.T.to_string())
    # print(per_target.T.head(n=100))


@main.command()
# @click.pass_context
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--input", "input_idx", type=int, help="Input index")
@click.option("--limit", "limit", type=int, help="Limit number of benchmark configs generated")
@click.option("--verbose", "verbose", type=bool, help="verbose output")
@click.option("--out", "output_path", help="Output path for combined stats")
def generate(path, config_path, bench_name, input_idx, limit, verbose, output_path):
    from pprint import pprint
    import wasabi

    benches = []

    b = Benchmarks(path)
    results_dir = Path(b.config["results_dir"])

    for target in [
        Target.Profile,
        Target.Simulate,
        Target.AccelsimSimulate,
        Target.PlaygroundSimulate,
    ]:
        if bench_name is None:
            for bench_configs in b.benchmarks[target.value].values():
                benches.extend(bench_configs)
        else:
            benches.extend(b.benchmarks[target.value][bench_name])

    if limit is not None:
        benches = benches[:limit]

    print(f"processing {len(benches)} benchmark configurations")

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    all_stats = []
    for bench_config in benches:
        # pprint(bench_config)
        name = bench_config["name"]
        target = bench_config["target"]
        input_idx = bench_config["input_idx"]
        input_values = bench_config["values"]

        match target.lower():
            case "profile":
                bench_stats = native.NsightStats(config, bench_config)
            case "simulate":
                if bench_config["values"]["mode"] != "serial":
                    continue
                bench_stats = stats.Stats(config, bench_config)
            case "accelsimsimulate":
                bench_stats = accelsim.Stats(config, bench_config)
            case "playgroundsimulate":
                bench_stats = playground.Stats(config, bench_config)
            case other:
                print(f"WARNING: {name} has unknown target {other}")
                continue

        print(f" ===> [{target}] \t\t {name}@{input_idx} \t\t {input_values}")

        values = pd.DataFrame.from_records([bench_config["values"]])
        values.columns = ["input_" + c for c in values.columns]
        # values.columns = [name + "input_" + c for c in values.columns]

        # this will be the new index
        values["target"] = target
        values["benchmark"] = name
        values["input_id"] = input_idx

        # print(bench_stats.result_df)
        # assert "run" in bench_stats.result_df.columns
        values = bench_stats.result_df.merge(values, how="cross")
        assert "run" in values.columns

        if verbose:
            print(values.T)
        all_stats.append(values)
        # print(bench_stats.result_df.T)

        # print("======")
        # print(bench_stats.print_all_stats())

    all_stats = pd.concat(all_stats)
    if verbose:
        print(all_stats)

    if bench_name is None:
        all_stats_output_path = results_dir / "combined.stats.csv"
    else:
        all_stats_output_path = results_dir / f"combined.stats.{bench_name}.csv"

    if output_path is not None:
        all_stats_output_path = Path(output_path)

    print(f"saving to {all_stats_output_path}")
    all_stats_output_path.parent.mkdir(parents=True, exist_ok=True)
    all_stats.to_csv(all_stats_output_path, index=False)

    return

    pprint(config)

    for bench_config in benches:
        name = bench_config["name"]
        input_idx = bench_config["input_idx"]
        print(f"\n\n=== {name}@{input_idx} ===")

        our_stats = stats.Stats(config, bench_config)
        playground_stats = playground.Stats(config, bench_config)
        accelsim_stats = accelsim.Stats(config, bench_config)
        native_stats = native.Stats(config, bench_config)

        # data = [
        #     ("native", native_stats.instructions(), accelsim_stats.instructions()),
        #     ("cycles", native_stats.cycles(), accelsim_stats.cycles()),
        # ]
        # print(
        #     wasabi.table(
        #         data,
        #         header=("", "instructions", "cycles"),
        #         divider=True,
        #         aligns=("r", "r", "r"),
        #     )
        # )

        data = [
            (
                "instructions",
                native_stats.instructions(),
                our_stats.instructions(),
                accelsim_stats.instructions(),
                playground_stats.instructions(),
            ),
            (
                "num blocks",
                native_stats.num_blocks(),
                our_stats.num_blocks(),
                accelsim_stats.num_blocks(),
                playground_stats.num_blocks(),
            ),
            (
                "warp instructions",
                native_stats.warp_instructions(),
                our_stats.warp_instructions(),
                accelsim_stats.warp_instructions(),
                playground_stats.warp_instructions(),
            ),
            (
                "cycles",
                native_stats.cycles(),
                our_stats.cycles(),
                accelsim_stats.cycles(),
                playground_stats.cycles(),
            ),
            (
                "exec time sec",
                native_stats.exec_time_sec(),
                our_stats.exec_time_sec(),
                accelsim_stats.exec_time_sec(),
                playground_stats.exec_time_sec(),
            ),
            (
                "dram reads",
                native_stats.dram_reads(),
                our_stats.dram_reads(),
                accelsim_stats.dram_reads(),
                playground_stats.dram_reads(),
            ),
            (
                "dram writes",
                native_stats.dram_writes(),
                our_stats.dram_writes(),
                accelsim_stats.dram_writes(),
                playground_stats.dram_writes(),
            ),
            (
                "dram accesses",
                native_stats.dram_accesses(),
                our_stats.dram_accesses(),
                accelsim_stats.dram_accesses(),
                playground_stats.dram_accesses(),
            ),
            (
                "L2 reads",
                native_stats.l2_reads(),
                our_stats.l2_reads() * 4,
                accelsim_stats.l2_reads(),
                playground_stats.l2_reads(),
            ),
            (
                "L2 writes",
                native_stats.l2_writes(),
                our_stats.l2_writes() * 4,
                accelsim_stats.l2_writes(),
                playground_stats.l2_writes(),
            ),
            (
                "L2 accesses",
                native_stats.l2_accesses(),
                our_stats.l2_accesses() * 4,
                accelsim_stats.l2_accesses(),
                playground_stats.l2_accesses(),
            ),
            (
                "L2 read hits",
                native_stats.l2_read_hits(),
                our_stats.l2_read_hits() * 4,
                accelsim_stats.l2_read_hits(),
                playground_stats.l2_read_hits(),
            ),
            (
                "L2 write hits",
                native_stats.l2_write_hits(),
                our_stats.l2_write_hits() * 4,
                accelsim_stats.l2_write_hits(),
                playground_stats.l2_write_hits(),
            ),
            (
                "L2 read misses",
                native_stats.l2_read_misses(),
                our_stats.l2_read_misses() * 4,
                accelsim_stats.l2_read_misses(),
                playground_stats.l2_read_misses(),
            ),
            (
                "L2 write misses",
                native_stats.l2_write_misses(),
                our_stats.l2_write_misses() * 4,
                accelsim_stats.l2_write_misses(),
                playground_stats.l2_write_misses(),
            ),
        ]
        data = [
            (
                k,
                human_readable(native),
                human_readable(ours),
                human_readable(accel),
                human_readable(play),
            )
            for (k, native, ours, accel, play) in data
        ]
        # print(native_stats.df)
        print(
            wasabi.table(
                data,
                header=("", "native", "ours", "accelsim", "playground"),
                divider=True,
                aligns=("r", "r", "r", "r", "r"),
            )
        )
        # , widths=widths, ))


if __name__ == "__main__":
    main()
    # main(ctx={})
