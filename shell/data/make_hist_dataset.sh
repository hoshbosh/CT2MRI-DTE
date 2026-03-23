CT_name="ct.nii"
MR_name="mr.nii"
HW=180
for which in "train" "valid" "test"
do
	# for plane in "axial" "sagittal" "coronal"
	for plane in "axial"
	do
	    for hist_type in "normal"
	    do
	    python -u ./brain_dataset_utils/generate_total_hist_global.py \
		    --plane $plane\
		    --hist_type $hist_type \
		    --which_set $which \
		    --height $HW \
		    --width $HW \
		    --pkl_name "/blue/neurology-dept/jlabasbas/pkls/MR_hist_global_${HW}_${which}_${plane}_$hist_typ.pkl" \
		    --data_dir "/blue/neurology-dept/jlabasbas/out-fine" \
		    --data_csv "/blue/neurology-dept/jlabasbas/out-fine/data.csv" \
		    --CT_name $CT_name \
		    --MR_name $MR_name \
		    > ./datasets/hdf5_log/MR_hist_global_${HW}_${which}_${plane}_$hist_type.log
	    done      
	done      
done
