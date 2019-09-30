rsync -av -e ssh --include='*.py' --exclude='*' --update . "$SCIAMA_URL:hierarchial_spiral_modelling"
rsync -av -e ssh --include='*.png' --exclude='*' --update "$SCIAMA_URL:hierarchial_spiral_modelling" .
rsync -av -e ssh --include='*.csv' --update "$SCIAMA_URL:hierarchial_spiral_modelling/saved_gzb_bhsm_trace" .
