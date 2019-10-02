# echo "Sending python scripts"
rsync -av --exclude=".git" --include "*/" --include="*.py" --exclude="*" . "$SCIAMA_URL:hierarchial_spiral_modelling"

# echo "Receiving plots"
# scp -r  "$SCIAMA_URL:hierarchial_spiral_modelling/plots" .
# scp -r  "$SCIAMA_URL:hierarchial_spiral_modelling/super_simple/plots/" .
#
# echo "Receiving chains"
# scp -r "$SCIAMA_URL:hierarchial_spiral_modelling/saved_gzb_bhsm_trace" .
