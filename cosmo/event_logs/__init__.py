from .reader import (
    bpi12,
    bpi13_incidents,
    bpi13_problems,
    bpi17,
    bpi19,
    # bpi20_domestic,
    # bpi20_international,
    bpi20_permit,
    bpi20_prepaid,
    bpi20_req4pay,
    sepsis,
)
from .reader import get_declare
from .as_dataset import ContinuousTraces, ConstrainedContinuousTraces


LOG_READERS = {
    # "bpi12": bpi12,
    "bpi13_incidents": bpi13_incidents,
    "bpi13_problems": bpi13_problems,
    "bpi17": bpi17,
    # "bpi19": bpi19,
    # "bpi20_domestic": bpi20_domestic,
    # "bpi20_international": bpi20_international,
    "bpi20_permit": bpi20_permit,
    "bpi20_prepaid": bpi20_prepaid,
    "bpi20_req4pay": bpi20_req4pay,
    "sepsis": sepsis,
}

__all__ = [
    # log readers
    "get_declare",
    "LOG_READERS",
    # dataset readers
    "ContinuousTraces",
    "ConstrainedContinuousTraces",
]
