from os import PathLike


class Stats:
    def __init__(self, result_dir: PathLike) -> None:
        self.path = result_dir
