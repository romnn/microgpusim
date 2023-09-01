import click
import yaml

import gpucachesim.stats.stats as stats
import gpucachesim.stats.native as native
from gpucachesim.benchmarks import Benchmarks, GPUConfig, REPO_ROOT_DIR


DEFAULT_CONFIG_FILE = REPO_ROOT_DIR / "./accelsim/gtx1080/gpgpusim.config.yml"


@click.command()
@click.option("--path", help="Path to materialized benchmark config")
@click.option("--config", default=DEFAULT_CONFIG_FILE, help="Path to GPU config")
@click.option("--bench", help="Benchmark name")
@click.option("--input", default=0, help="Input index")
def main(path, config, bench, input):
    from pprint import pprint

    b = Benchmarks(path)
    if bench is None:
        raise NotImplemented
    print(bench, input)
    bench_config = b.get_bench_config(bench, input)
    # pprint(bench_config)

    with open(config, "rb") as f:
        config: GPUConfig = yaml.safe_load(f)

    pprint(config)
    our_stats = stats.Stats(bench_config)
    native_stats = native.Stats(config, bench_config)

    print(native_stats.cycles())
    print(our_stats.cycles())


if __name__ == "__main__":
    main()
