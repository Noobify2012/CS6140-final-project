"""loader.py
Load files from various sources to dataframes for CS6140 final project.
"""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from enum import Enum
from pathlib import Path
from typing import List


# kaggle_zip = "https://storage.googleapis.com/kaggle-data-sets/2529204/4295427/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230425%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230425T000637Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=098f331678d01df0f8f650327f0b982df5d86e95a1785c3c0f6f9c9625bfe043f7a4fa3d78d4cacb6b2d154892783cc223727353b3c7f654123cd576c04d3e9f96b2f1801674f1fb589d3810519ff361e78dc2b6d094fb423ab9e35777fc908962319b923e7819bdde19ec1720e82b665daca8169e3e2554d1cc0996057c1ee01cd2456be518532a42a060ff109f73720e50afdeaacb78cfc1c4afaec41b4964deca035bf33e08055e4ee52b902bc26fc2d923a31ab491ac54df9681271d53a25abf56451ff9be8f6492280cb5508632fbf20a0a9a436bcc2a066e072f9da1e39b1f69af90c6ab3c282c9626f56f7e84b9e85703e1c4a8cfa410849e28b09bb4"
# raw_zip = "https://storage.googleapis.com/kaggle-data-sets/2529204/4295427/compressed/raw.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230425%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230425T003646Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1da714c2b106d3b15b5d16fafd617120948bdc295d17d7a1dd3d6840299a86ace965b81496e69a161d0c2bd56e9d3fe510e2192eec353c34af96777d19a7f2abfe997332c34c988899f9fba27b08701c3c59768e1ee9c9aa9ffed5084eeb6235479ba30c4cbaf4b51f2a9f6d72faf60dabec8ca1cb193571a77e66aba125e7154133d2bfd753e370813fd9707922cee85e4c808e218c0700e3279707053e7d6954566fa673bd1021d114fbb44aecaa85253a7d1d4e8325dc2b25c585cbb830fd18f5d2067c9bc5d4b8a36f3712176bc46068a35a7cd91a38625afd82ffd2d81d46ce354761188457c25fd801b12e7d35cf273fd8a3478c232d54678c7e0bdc6d"
# dataset_id = 2529204
# dataset_user = "robikscube"
# dataset_name = "flight-delay-dataset-20182022"

res_dir = Path.cwd().parent / "res"
all_pickle_file = "df_all.pkl"

drop_columns = [
    "Quarter",
    "FlightDate",
    "Marketing_Airline_Network",
    "Marketing_Airline_Network",
    "Operated_or_Branded_Code_Share_Partners",
    "DOT_ID_Marketing_Airline",
    "IATA_Code_Marketing_Airline",
    "Flight_Number_Marketing_Airline",
    "Originally_Scheduled_Code_Share_Airline",
    "DOT_ID_Originally_Scheduled_Code_Share_Airline",
    "IATA_Code_Originally_Scheduled_Code_Share_Airline",
    "Flight_Num_Originally_Scheduled_Code_Share_Airline",
    "DOT_ID_Operating_Airline",
    "IATA_Code_Operating_Airline",
    "Flight_Number_Operating_Airline",
    "OriginAirportID",
    "OriginAirportSeqID",
    "OriginCityMarketID",
    "OriginCityName",
    "OriginState",
    "OriginStateFips",
    "OriginStateName",
    "OriginWac",
    "DestAirportID",
    "DestAirportSeqID",
    "DestCityMarketID",
    "DestCityName",
    "DestState",
    "DestStateFips",
    "DestStateName",
    "DestWac",
    "CRSDepTime",
    "DepDelay",
    "DepDelayMinutes",
    "DepDel15",
    "DepartureDelayGroups",
    "DepTimeBlk",
    "TaxiOut",
    "WheelsOff",
    "WheelsOn",
    "TaxiIn",
    "CRSArrTime",
    "ArrTime",
    "ArrDelay",
    "ArrDelayMinutes",
    "ArrivalDelayGroups",
    "ArrTimeBlk",
    "Cancelled",
    "CancellationCode",
    "Diverted",
    "CRSElapsedTime",
    "ActualElapsedTime",
    "AirTime",
    "Flights",
    "Distance",
    "FirstDepTime",
    "TotalAddGTime",
    "LongestAddGTime",
    "DivAirportLandings",
    "DivReachedDest",
    "DivActualElapsedTime",
    "DivArrDelay",
    "DivDistance",
    "Div1Airport",
    "Div1AirportID",
    "Div1AirportSeqID",
    "Div1WheelsOn",
    "Div1TotalGTime",
    "Div1LongestGTime",
    "Div1WheelsOff",
    "Div1TailNum",
    "Div2Airport",
    "Div2AirportID",
    "Div2AirportSeqID",
    "Div2WheelsOn",
    "Div2TotalGTime",
    "Div2LongestGTime",
    "Div2WheelsOff",
    "Div2TailNum",
    "Div3Airport",
    "Div3AirportID",
    "Div3AirportSeqID",
    "Div3WheelsOn",
    "Div3TotalGTime",
    "Div3LongestGTime",
    "Div3WheelsOff",
    "Div3TailNum",
    "Div4Airport",
    "Div4AirportID",
    "Div4AirportSeqID",
    "Div4WheelsOn",
    "Div4TotalGTime",
    "Div4LongestGTime",
    "Div4WheelsOff",
    "Div4TailNum",
    "Div5Airport",
    "Div5AirportID",
    "Div5AirportSeqID",
    "Div5WheelsOn",
    "Div5TotalGTime",
    "Div5LongestGTime",
    "Div5WheelsOff",
    "Div5TailNum",
    # "Duplicate",
    "Unnamed: 119",
]


class FileSourceEnum(Enum):
    """Simple enum to help differentiate when running on kaggle or local."""

    LOCAL = ("local", Path.cwd().parent / "raw")
    KAGGLE = ("kaggle", Path("/kaggle/input"))

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
    """Returns the FileSoureEnum based on the location. Errors if neither local
    or kaggle is found.
    """
    if FileSourceEnum.KAGGLE.path.exists():
        return FileSourceEnum.KAGGLE
    elif FileSourceEnum.LOCAL.path.exists():
        return FileSourceEnum.LOCAL
    else:
        raise FileNotFoundError(
            f"couldn't find Kaggle files or local files "
            + "in {FileSourceEnum.LOCAL.path}"
        )


# def get_file_generator() -> Generator:
#     """Create a simple generate for the files found in the FileSourceEnum Path."""
#     for child in get_location().path.iterdir():
#         if child.is_dir():
#             yield from get_file_generator(child)
#         else:
#             yield child


def _get_csv_by_name(name: str) -> Path:
    """Get a single CSV file by name"""
    files = list(get_location().path.rglob(name))
    if len(files) > 0:
        return files[0]
    raise ValueError("Can't find file")


def _get_csv_by_year(year: int) -> List[Path]:
    """Get a list of CSV files by year"""
    return list(get_location().path.rglob(f"Flights_{year}_*.csv"))


def _get_all_csv_files() -> List[Path]:
    """Get all the CSV files"""
    return list(get_location().path.rglob("*.csv"))


def _get_df_from_csv(
    *,
    year: int = None,
    file: str = None,
    all_files: bool = False,
    drop_columns=drop_columns,
) -> pd.DataFrame:
    """return a pandas dataframe based on the provided stats"""

    if all(e is None for e in [file, year]) and not all_files:
        raise ValueError("You need at least one parameter dude")
    if file:
        file = _get_csv_by_name(file)
        df = pd.read_csv(file, low_memory=False)
        df = df.drop(columns=drop_columns)
    else:
        df = pd.DataFrame()
        files = _get_all_csv_files() if all_files else _get_csv_by_year(year)
        for file in tqdm(files):
            temp_df = pd.read_csv(file, low_memory=False)
            temp_df = temp_df.drop(columns=drop_columns)
            df = pd.concat([df, temp_df], ignore_index=True)
    return df.rename(columns={"Operating_Airline ": "Operating_Airline"})


def _save_df(name: str, df: pd.DataFrame) -> None:
    if get_location() != FileSourceEnum.LOCAL:
        raise ValueError("Can't save when not running locally")

    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(res_dir.resolve() / name)


def _load_pickle(file: str) -> pd.DataFrame:
    if get_location() == FileSourceEnum.LOCAL:
        return pd.read_pickle(res_dir / file)
    raise ValueError("Not local bro!")


def get_df(
    drop_columns: List[str] =drop_columns,
    *,
    year: int = None,
    file: str = None,
    all_files: bool = False,
) -> pd.DataFrame:
    """Returns a Pandas DataFrame with data for a specific year and/or file.


    :param year: The year to filter the data by. Defaults to None.
    :type year: int, optional
    :param file: The name of the file to load data from. Defaults to None.
    :type file: str, optional
    :param all_files: Whether to load data from all available files.
                      Defaults to False.
    :type all_files: bool, optional
    :return: A Pandas DataFrame with the filtered data.
    :rtype: pd.DataFrame
    :raises ValueError: If no arguments are provided
    """

    if all(e is None for e in [file, year]) and not all_files:
        raise ValueError("You need at least one parameter dude")

    if all_files:
        if (res_dir / all_pickle_file).exists():
            return _load_pickle(res_dir / all_pickle_file)
        else:
            df = _get_df_from_csv(all_files=True, drop_columns=drop_columns)
            _save_df(all_pickle_file, df)
            return df
    elif year is not None:
        if (res_dir / f"df_{year}.pkl").exists():
            return _load_pickle(res_dir / f"df_{year}.pkl")
        else:
            df = _get_df_from_csv(year=year, drop_columns=drop_columns)
            _save_df(f"df_{year}.pkl", df)
            return df
    else:
        parts = file.split("_")
        back_end = parts[1:]
        if len(back_end) > 1:
            year_part = back_end[0]
            split = back_end[1].split(".")
            num_part = "_" + split[0]
        else:
            year_part = back_end[0].split(".")[0]
            num_part = ""
        file_name = f"df_{year_part}{num_part}.pkl"
        if (res_dir / file_name).exists():
            return _load_pickle(file_name)
        else:
            if num_part == "":
                df = _get_df_from_csv(
                    year=year_part, drop_columns=drop_columns
                )
            else:
                df = _get_df_from_csv(
                    file=f"Flights_{year_part}{num_part}.csv",
                    drop_columns=drop_columns,
                )
            _save_df(file_name, df)
            return df
