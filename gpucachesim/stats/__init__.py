import click
import yaml

import gpucachesim.stats.stats as stats
import gpucachesim.stats.native as native
import gpucachesim.stats.accelsim as accelsim
import gpucachesim.stats.playground as playground
from gpucachesim.benchmarks import Benchmarks, GPUConfig, REPO_ROOT_DIR


DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.config.yml"


def human_readable(n: int) -> str:
    return "{:,}".format(n).replace(",", " ")


@click.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--config", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", "bench_name", help="Benchmark name")
@click.option("--input", "input_idx", type=int, help="Input index")
def main(path, config, bench_name, input_idx):
    from pprint import pprint
    import wasabi

    benches = []

    b = Benchmarks(path)
    if bench_name is None:
        raise NotImplemented

    if input_idx is None:
        benches.extend(b[bench_name])
    else:
        benches.append(b.get_bench_config(bench_name, input_idx))

    with open(config, "rb") as f:
        config = GPUConfig(yaml.safe_load(f))

    pprint(config)

    for bench_config in benches:
        name = bench_config["name"]
        input_idx = bench_config["input_idx"]
        print(f"\n\n=== {name}@{input_idx} ===")

        # our_stats = stats.Stats(bench_config)
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
                accelsim_stats.instructions(),
                playground_stats.instructions(),
            ),
            (
                "cycles",
                native_stats.cycles(),
                accelsim_stats.cycles(),
                playground_stats.cycles(),
            ),
            (
                "dram reads",
                native_stats.dram_reads(),
                accelsim_stats.dram_reads(),
                playground_stats.dram_reads(),
            ),
            (
                "dram writes",
                native_stats.dram_writes(),
                accelsim_stats.dram_writes(),
                playground_stats.dram_writes(),
            ),
            (
                "dram accesses",
                native_stats.dram_accesses(),
                accelsim_stats.dram_accesses(),
                playground_stats.dram_accesses(),
            ),
            (
                "L2 reads",
                native_stats.l2_reads(),
                accelsim_stats.l2_reads(),
                playground_stats.l2_reads(),
            ),
            (
                "L2 writes",
                native_stats.l2_writes(),
                accelsim_stats.l2_writes(),
                playground_stats.l2_writes(),
            ),
            (
                "L2 accesses",
                native_stats.l2_accesses(),
                accelsim_stats.l2_accesses(),
                playground_stats.l2_accesses(),
            ),
        ]
        data = [
            (k, human_readable(native), human_readable(accel), human_readable(play))
            for (k, native, accel, play) in data
        ]
        print(native_stats.df)
        print(
            wasabi.table(
                data,
                header=("", "native", "accelsim", "playground"),
                divider=True,
                aligns=("r", "r", "r", "r"),
            )
        )
        # , widths=widths, ))


if __name__ == "__main__":
    main()
