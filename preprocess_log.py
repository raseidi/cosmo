""" 
1st we gotta preprocess data
then extract the declare rules
so everytime we change preprocessing 
we need to re-extract declare rules
"""

import os
import pm4py
import argparse
import pandas as pd

from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.Encodings.Declare import Declare

from cosmo.utils import ensure_dir
from cosmo.event_logs.utils import clear_cache
from cosmo.event_logs import LOG_READERS, get_declare

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-name", type=str, default="bpi13_problems")
    parser.add_argument("--overwrite", type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    out_path = os.path.join("data", args.log_name, "declare", "constraints.pkl")

    # if args.overwrite:
    #     clear_cache(args.log_name)

    log_reader = LOG_READERS.get(args.log_name, None)
    if log_reader is None:
        raise ValueError(f"Dataset {args.log_name} not found")
    log = log_reader()
    log = log.rename(columns={"case_id": "case:concept:name",
                              "activity": "concept:name",
                              "timestamp": "time:timestamp"})
    log["time:timestamp"] = pd.to_datetime(log["time:timestamp"], infer_datetime_format=True)
    # convert to pm4py.EventLog to use Declare4Py
    
    # idk why some dfs have attrs others do not
    log.attrs = {'pm4py:param:activity_key': 'concept:name', 'pm4py:param:attribute_key': 'concept:name', 'pm4py:param:timestamp_key': 'time:timestamp', 'pm4py:param:resource_key': 'org:resource', 'pm4py:param:transition_key': 'lifecycle:transition', 'pm4py:param:group_key': 'org:group'}
    if os.path.exists(out_path):
        print("Declare rules already extracted")
        exit(0)

    log = pm4py.convert_to_event_log(log[log["concept:name"] != "<EOS>"])
    log = D4PyEventLog(log=log)

    encoder = Declare(log)
    encoded = encoder.transform(log)
    encoded.reset_index(inplace=True)
    encoded = encoded.rename(columns={"index": "case_id"})
    encoded.to_pickle(out_path)
