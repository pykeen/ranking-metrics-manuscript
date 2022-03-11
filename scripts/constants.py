from pathlib import Path

HERE = Path(__file__).parent.parent.resolve()
MODEL_DIRECTORY = HERE.joinpath("glb_results")

COLLATION_DIRECTORY = HERE.joinpath("collated")
COLLATION_DIRECTORY.mkdir(exist_ok=True, parents=True)

COLLATED_PATH = COLLATION_DIRECTORY.joinpath("collated.tsv")
MELTED_PATH = COLLATION_DIRECTORY.joinpath("melted.tsv")

CHARTS_DIRECTORY = HERE.joinpath("charts")
CHARTS_DIRECTORY.mkdir(exist_ok=True, parents=True)

DEFAULT_CACHE_DIRECTORY = HERE.joinpath("macro_cache")
