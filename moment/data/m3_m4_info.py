import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass
class M3Yearly:
    seasonality: int = 1
    horizon: int = 6
    freq: str = "Y"
    sheet_name: str = "M3Year"
    name: str = "Yearly"
    n_ts: int = 645


@dataclass
class M3Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = "Q"
    sheet_name: str = "M3Quart"
    name: str = "Quarterly"
    n_ts: int = 756


@dataclass
class M3Monthly:
    seasonality: int = 12
    horizon: int = 18
    freq: str = "M"
    sheet_name: str = "M3Month"
    name: str = "Monthly"
    n_ts: int = 1428


@dataclass
class M3Other:
    seasonality: int = 1
    horizon: int = 8
    freq: str = "D"
    sheet_name: str = "M3Other"
    name: str = "Other"
    n_ts: int = 174


@dataclass
class M4Yearly:
    seasonality: int = 1
    horizon: int = 6
    freq: str = "Y"
    name: str = "Yearly"
    n_ts: int = 23_000


@dataclass
class Quarterly:
    seasonality: int = 4
    horizon: int = 8
    freq: str = "Q"
    name: str = "Quarterly"
    n_ts: int = 24_000


@dataclass
class M4Monthly:
    seasonality: int = 12
    horizon: int = 18
    freq: str = "M"
    name: str = "Monthly"
    n_ts: int = 48_000


@dataclass
class M4Weekly:
    seasonality: int = 1
    horizon: int = 13
    freq: str = "W"
    name: str = "Weekly"
    n_ts: int = 359


@dataclass
class M4Daily:
    seasonality: int = 1
    horizon: int = 14
    freq: str = "D"
    name: str = "Daily"
    n_ts: int = 4_227


@dataclass
class M4Hourly:
    seasonality: int = 24
    horizon: int = 48
    freq: str = "H"
    name: str = "Hourly"
    n_ts: int = 414


@dataclass
class M4Other:
    seasonality: int = 1
    horizon: int = 8
    freq: str = "D"
    name: str = "Other"
    n_ts: int = 5_000
    included_groups: Tuple = ("Weekly", "Daily", "Hourly")
