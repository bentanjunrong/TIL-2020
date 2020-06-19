# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import pandas as pd
from pathlib import Path
import json
import collections
import numpy as np


def json_to_pd(truth_path,ans_path):
    truth_json_path = Path(truth_path)
    truth_annot_json = json.load(truth_json_path.open())
    truth_annot = pd.read_json(json.dumps(truth_annot_json['annotations']))

    ans_json_path = Path(ans_path)
    ans_json = json.load(ans_json_path.open())
    ans_annot = pd.read_json(json.dumps(ans_json))

    truth_annot = truth_annot.sort_values(by=['image_id','category_id'],ignore_index=True)
    ans_annot = ans_annot.sort_values(['image_id','category_id'],ignore_index=True)

    truth_annot['detected'] = False

    truth_annot.drop(columns=['area','iscrowd','id'], inplace=True)


    # truth_annot_grouped = truth_annot.groupby(['image_id','category_id'])

    return truth_annot, ans_annot










