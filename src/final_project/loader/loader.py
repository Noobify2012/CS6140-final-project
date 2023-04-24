"""loader.py
Load files from various sources to dataframes for CS6140 final project.
"""

from __future__ import annotations
from dotenv import load_dotenv
from enum import Enum
from pathlib import Path
from typing import Generator



class FileSourceEnum(Enum):
    LOCAL = ('local', Path.cwd().parent / 'raw')
    KAGGLE = ('kaggle', Path("/kaggle/input"))

    def __init__(self, title: str, path: Path):
        self._title = title
        self._path = path

    @property
    def title(self) -> str:
        return self._title

    @property
    def path(self) -> Path:
        return self._path


def get_location() -> FileSourceEnum:
    if FileSourceEnum.KAGGLE.path.exists():
        return FileSourceEnum.KAGGLE
    elif FileSourceEnum.LOCAL.path.exists():
        return FileSourceEnum.LOCAL
    else:
        raise FileNotFoundError(f"couldn't find Kaggle files or local files in {FileSourceEnum.LOCAL.path}")


def get_file_generator() -> Generator:
    for child in get_location().path.iterdir():
        if child.is_dir():
            yield from get_file_generator(child)
        else:
            yield child

