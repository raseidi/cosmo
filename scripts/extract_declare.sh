declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi17" "bpi19" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")

for dataset in "${DATASETS[@]}"
do
    echo "Extracting declare for $dataset"
    python preprocess_log.py --log-name $dataset
done