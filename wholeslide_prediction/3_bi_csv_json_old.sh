SLIDENAME=BC_065_0_1
SLIDEPATH=/data01/shared/tcga_analysis/seer_data/images/Hawaii/batch3/BC_065_0_1.svs
nohup python -u binarize_pred.py ${SLIDENAME} >log.bi.${SLIDENAME}.txt &
nohup python jason_new.py ${SLIDEPATH} &
