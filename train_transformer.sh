#bin/bash

LR=(0.005 0.0005 0.00005)
N_HEADS=(1 2 4 8)
LAYERS=(1)
INPUT_SIZES=(16)
HIDDEN_SIZES=(32)
declare -a BACKBONES=("mha" "transformer")
declare -a TEMPLATES=("existence" "choice" "positive relations")

PROJECT="cosmo-mha"

declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi17" "bpi19" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")

for dataset in "${DATASETS[@]}"
do
    echo "Extracting declare for $dataset"
    python preprocess_log.py --log-name $dataset
done

python get_experiments.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do
    for backbone in "${BACKBONES[@]}"
    do
        for lr in "${LR[@]}"
        do
            for n_heads in "${N_HEADS[@]}"
            do
                echo "Training for $dataset with lr=$lr, batch_size=$batch_size, hidden_size=$hidden_size"
                python train.py \
                --dataset $dataset \
                --backbone $backbone \
                --lr $lr \
                --n-heads $n_heads \
                --wandb True \
                --project-name $PROJECT
            done
        done
    done
done