import re
import os
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
from gzbuilder_analysis import load_aggregation_results, load_fit_results

agg_res_path = '/Users/tlingard/PhD/gzbuilder_collection/gzbuilder_results/output_files/aggregation_results'
fit_model_path = '/Users/tlingard/PhD/gzbuilder_collection/gzbuilder_results/output_files/tuning_results'

agg_results = load_aggregation_results(agg_res_path)
fit_results = load_fit_results(fit_model_path)

arms_df = pd.Series([], dtype=object)
with tqdm(fit_results.index, desc='Correcting disks') as bar:
    for subject_id in bar:
        fit_disk = fit_results.loc[subject_id]['fit_model'].disk
        spirals = agg_results.loc[subject_id].spiral_arms
        arms = pd.Series([], dtype=object)
        for i in range(len(spirals)):
            arm = deepcopy(spirals[i])
            arm.modify_disk(
                centre=fit_disk[['mux', 'muy']],
                phi=fit_disk.roll,
                ba=fit_disk.q,
            )
            arms[i] = arm
        try:
            arms['pipeline'] = arms.loc[0].get_parent()
        except KeyError:
            arms['pipeline'] = None
        arms_df[subject_id] = arms

arms_df.apply(pd.Series).to_pickle('lib/spiral_arms2.pickle')
