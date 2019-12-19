import numpy as np
import pandas as pd
from tqdm import tqdm

arms = pd.read_pickle('lib/unweighted_spirals.pickle')
duplicates = pd.read_csv('lib/duplicate_galaxies.csv', index_col=0)
duplicates = duplicates.rename(
    columns={'0': 'original', '1': 'validation'}
).astype(int)

original_arms = arms.loc[duplicates['original'].values]
validation_arms = arms.loc[duplicates['validation'].values]\
    .drop(columns='pipeline')\
    .rename(columns=lambda c: f'val-{c}')
validation_arms.index = original_arms.index

combined_arms = pd.concat((original_arms, validation_arms), axis=1)
in_val = np.logical_or(np.isin(arms.index, duplicates['original']), np.isin(arms.index, duplicates['validation']))
merged_arms = pd.Series([])
with tqdm(combined_arms.index.values) as bar:
    for idx in bar:
        res = combined_arms.loc[idx]['pipeline'].merge_arms(
            combined_arms.loc[idx].drop('pipeline').dropna().values
        )
        merged_arms.loc[idx] = dict(
            pipeline=combined_arms.loc[idx]['pipeline'],
            **{f'arm_{i}': j for i, j in enumerate(res)}
        )

final_arms = pd.concat((
    merged_arms.apply(pd.Series),
    arms[~in_val],
), axis=0)
print(final_arms)
final_arms.to_pickle('lib/merged_arms.pickle')
