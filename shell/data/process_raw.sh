# 1. Unzip raw file
DATA_DIR=$1
unzip $DATA_DIR
# 2. Delete unneeded pelvis scans
rm -rf $DATA_DIR/pevlis
# 3. Iterate through each subject and unzip and rename nii files,

ORIG_ROOT="./brain/"         # Change to your original dataset root
COPY_ROOT="${ORIG_ROOT}_copy"         # The copy will be named root_dir_copy
CSV_FILE="$COPY_ROOT/data.csv"

cp -r "$ORIG_ROOT" "$COPY_ROOT"

subjects=()
for dir in "$COPY_ROOT"/*(/); do
    pid=$(basename "$dir")
    # Unzip and then delete CT.nii.gz and MR.nii.gz if they exist
    if [[ -f "$dir/ct.nii.gz" ]]; then
        gunzip -k "$dir/ct.nii.gz" && rm "$dir/ct.nii.gz"
    fi
    if [[ -f "$dir/mr.nii.gz" ]]; then
        gunzip -k "$dir/mr.nii.gz" && rm "$dir/mr.nii.gz"
    fi
    rm "$dir/mask.nii.gz"
    subjects+=("$pid")
done

# Shuffle subjects using sort and random
shuffled_subjects=("${(@f)$(printf "%s\n" "${subjects[@]}" | sort -R)}")

# Calculate split index
total=${#shuffled_subjects[@]}
split=$((total * 7 / 10))

# Write CSV header
echo "pid,set" > "$CSV_FILE"

# Write train set
for ((i=1; i<=split; i++)); do
    echo "${shuffled_subjects[$i]},train" >> "$CSV_FILE"
done

# Write test set
for ((i=split+1; i<=total; i++)); do
    echo "${shuffled_subjects[$i]},test" >> "$CSV_FILE"
done

echo "Copy created at $COPY_ROOT"
echo "CSV written to $CSV_FILE"

# 5. Resample nii files to desired resolutions
python ../../resize_all.py
# 6. Run create hdf5 file
./make_hdf5.sh
# 7. Profit??
