from os import PathLike

import gpucachesim.stats.common as common


class Stats(common.Stats):
    def __init__(self, result_dir: PathLike) -> None:
        self.path = result_dir

    def cycles(self) -> int:
        raise NotImplemented
