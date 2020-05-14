import re
import os
import numpy as np
from time import sleep
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

arms_df = arms_df.apply(pd.Series)
arms_df.to_pickle('lib/spiral_arms.pickle')


# Merge original and validation subsets

duplicates = pd.read_csv('lib/duplicate_galaxies.csv', index_col=0).rename(
    columns={'0': 'original', '1': 'validation'}
).astype(int)

combined_df = pd.concat((
    arms_df.loc[duplicates['original'].values].reset_index(drop=True),
    arms_df.loc[duplicates['validation'].values].reset_index(drop=True).add_prefix('val-'),
), axis=1)
combined_df.index = duplicates.index

reduced_combined_df = pd.concat((
    combined_df.apply(
        lambda a: a.pipeline if a.pipeline is not None else a['val-pipeline'],
        axis=1
    ).rename('pipeline'),
    combined_df.drop(columns=['pipeline', 'val-pipeline']).apply(
        lambda a: a.dropna().reset_index(drop=True),
        axis=1
    ).add_prefix('arm-')
), axis=1)

merged_arms = pd.Series(index=reduced_combined_df.index, dtype=object)
with tqdm(reduced_combined_df.iterrows(), total=len(reduced_combined_df)) as bar:
    for idx, row in bar:
        if row.pipeline is not None:
            arms = row.loc[[c for c in row.dropna().index if 'arm' in c]]
            __merged_arms = row.pipeline.merge_arms(arms)
            merged_arms[idx] = dict(
                pipeline=row.pipeline,
                **{
                    f'arm-{i}': __merged_arms[i]
                    for i in range(len(__merged_arms))
                }
            )
        else:
            sleep(0.1)

merged_spirals = pd.concat((
    merged_arms.dropna().apply(pd.Series).assign(subject_id=duplicates.original)
        .set_index('subject_id'),
    arms_df[
        ~np.isin(arms_df.index, np.unique(duplicates.values.ravel()))
    ].rename(columns={i: f'arm-{i}' for i in range(10)}),
), axis=0)

merged_spirals.to_pickle('lib/merged_arms.pickle')
