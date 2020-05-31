from utils import constants
import pathlib

CURRENT_PATH = pathlib.Path.cwd()

SPOP_TRUE_ORIGIN_DATA_PATH = CURRENT_PATH.parent / "data" / "origin" / constants.SPOP_TRUE
SPOP_FALSE_ORIGIN_DATA_PATH = CURRENT_PATH.parent / "data" / "origin" / constants.SPOP_FALSE


SPLITS_DATA_PATH = CURRENT_PATH.parent / "data" / "splits"
