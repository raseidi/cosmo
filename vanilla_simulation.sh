#bin/bash

declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")
declare -a TEMPLATES=("all")

for dataset in "${DATASETS[@]}"
do
    for template in "${TEMPLATES[@]}"
    do
        python simulation_vanilla.py --dataset $dataset --template "$template"
    done
done
