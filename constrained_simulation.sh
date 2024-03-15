#bin/bash

# declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay" "bpi17" "bpi19" )
declare -a DATASETS=("bpi17")
# declare -a TEMPLATES=("existence" "choice" "positive relations")
declare -a TEMPLATES=("existence" "choice" "positive relations")

for dataset in "${DATASETS[@]}"
do
    for template in "${TEMPLATES[@]}"
    do
        python simulation_crnn.py --dataset $dataset --template "$template"
    done
done
