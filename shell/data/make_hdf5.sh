
mkdir ./datasets/hdf5_log
for which in "train" "valid" "test"
do
    # for plane in "axial" "sagittal" "coronal"
    for plane in "axial"
    do
    touch ./datasets/hdf5_log/180_${which}_${plane}.log
    python -u brain_dataset_utils/generate_total_hdf5_csv.py \
            --plane  $plane\
            --which_set $which \
            --height 180 \
            --width 180 \
            --hdf5_name "/blue/neurology-dept/jlabasbas/hdf5s/180_${which}_${plane}.hdf5" \
            --data_dir "/blue/neurology-dept/jlabasbas/out" \
            --data_csv "/blue/neurology-dept/jlabasbas/out/data.csv" \
            --CT_name "ct.nii" \
            --MR_name "mr.nii" \
            > ./datasets/hdf5_log/180_${which}_${plane}.log
    done      
done
