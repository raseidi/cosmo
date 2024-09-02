#bin/bash

declare -a DATASETS=("sepsis" "bpi12" "bpi13_problems" "bpi20_permit" "bpi17")
declare -a TEMPLATES=("existence" "choice" "positive relations")

for dataset in "${DATASETS[@]}"
do
    for template in "${TEMPLATES[@]}"
    do
        python simulation_crnn.py --dataset $dataset --template "$template"
    done
done
