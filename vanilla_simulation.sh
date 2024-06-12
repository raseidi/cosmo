#bin/bash

declare -a DATASETS=("sepsis" "bpi12" "bpi13_problems" "bpi20_permit" "bpi17")
declare -a TEMPLATES=("all")

for dataset in "${DATASETS[@]}"
do
    for template in "${TEMPLATES[@]}"
    do
        python simulation_vanilla.py --dataset $dataset --template "$template"
    done
done
