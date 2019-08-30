import re
from astropy.io import fits
import numpy as np
import pandas as pd
import lib.galaxy_utilities as gu

dr7ids = gu.metadata['SDSS dr7 id'].dropna().astype(np.int64)
gz_fits = fits.open('../source_files/NSA_GalaxyZoo.fits')
gz_objids = gz_fits[1].data['dr7objid']
gz_bars = gz_fits[1].data['t03_bar_a06_bar_debiased']

gz_keys = [
    i for i in gz_fits[1].data.dtype.names if re.match(r't[0-9]+_', i)
]
gz_data = pd.DataFrame({
    k: gz_fits[1].data[k].byteswap().newbyteorder()
    for k in gz_keys
}, index=gz_objids)

gz_data_matched = gz_data.reindex(dr7ids.values)
gz_data_matched['subject_id'] = dr7ids.index.values
gz_data_matched = gz_data_matched.reset_index().set_index('subject_id')

gz_data_matched.to_csv('gzb_gz2_data.csv')
