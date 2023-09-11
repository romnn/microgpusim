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
from gpucachesim.benchmarks import Target, Benchmarks, GPUConfig, REPO_ROOT_DIR


DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.config.yml"


def human_readable(n) -> str:
    # return "{:}".format(n)

    import math

    if abs(float(int(n)) - float(n)) == 0:
        n = int(n)

    precision = 5
    if isinstance(n, float):
        res = "{:.5f}".format(n)
    else:
        res = "{}".format(n)
    # before the point
    parts = res.split(".")
    before, after = parts[0], "".join(parts[1:])

    num_segments = int(math.ceil(float(len(before)) / 3.0))
    before = reversed([before[max(len(before) - (3 * i) - 3, 0) : len(before) - (3 * i)] for i in range(num_segments)])
    before = " ".join(before)
    if after == "":
        # print(before.replace(" ", ""))
        # print(f"{round(n, precision):f}")
        assert float(before.replace(" ", "")) == round(n, precision)
        return before

    # print((before + "." + after).replace(" ", ""))
    # print(f"{round(n, precision):f}")
    assert float((before + "." + after).replace(" ", "")) == round(n, precision)
    return before + "." + after
    # return "{:,}".format(n).replace(",", " ")


@click.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--config", "config_path", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--input", "input_idx", type=int, help="Input index")
@click.option("--out", "output_path", help="Output path for combined stats")
def main(path, config_path, bench_name, input_idx, output_path):
    from pprint import pprint
    import wasabi

    benches = []

    b = Benchmarks(path)
    results_dir = Path(b.config["results_dir"])

    if bench_name is None:
        raise NotImplemented

    for target in [
        Target.Simulate,
        Target.Profile,
        Target.AccelsimSimulate,
        Target.PlaygroundSimulate,
    ]:
        benches.extend(b.benchmarks[target.value][bench_name])
        # for target_benches in b.benchmarks[target.value][bench_name]:
        #     pprint(target_benches)
        #     benches.extend(target_benches[bench_name])

    # if input_idx is None:
    #     benches.extend(b.get_bench_configs(bench_name))
    # else:
    #     benches.append(b.get_bench_config(bench_name, input_idx))

    print(len(benches))

    with open(config_path, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    all_stats = []
    for bench_config in benches:
        # pprint(bench_config)
        name = bench_config["name"]
        target = bench_config["target"]
        input_idx = bench_config["input_idx"]
        input_values = bench_config["values"]

        print(f" ===> [{target}] \t\t {name}@{input_idx} \t\t {input_values}")
        match target.lower():
            case "profile":
                bench_stats = native.Stats(config, bench_config)
            case "simulate":
                bench_stats = stats.Stats(config, bench_config)
            case "accelsimsimulate":
                bench_stats = accelsim.Stats(config, bench_config)
            case "playgroundsimulate":
                bench_stats = playground.Stats(config, bench_config)
            case other:
                print(f"WARNING: {name} has unknown target {other}")
                continue

        values = pd.DataFrame.from_records([bench_config["values"]])
        values.columns = ["input_" + c for c in values.columns]
        # values.columns = [name + "input_" + c for c in values.columns]

        # this will be the new index
        values["target"] = target
        values["benchmark"] = name
        values["input_id"] = input_idx

        values = bench_stats.result_df.merge(values, how="cross")

        # print("======")
        # print(values.T)
        all_stats.append(values)
        # print(bench_stats.result_df.T)

        # print("======")
        # print(bench_stats.print_all_stats())

    all_stats = pd.concat(all_stats)
    print(all_stats)

    all_stats_output_path = results_dir / "combined.stats.csv"
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
