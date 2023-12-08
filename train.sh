#bin/bash

LR=(0.0005)
BATCH_SIZES=(16)
EPOCHS=(50)
LAYERS=(1)
INPUT_SIZES=(32)
HIDDEN_SIZES=(256)
# declare -a TEMPLATES=("all" "existence" "choice" "positive relations" "negative relations")
declare -a TEMPLATES=("positive relations")

PROJECT="cosmo-ltl"


# declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi17" "bpi19" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")
declare -a DATASETS=("bpi20_permit")


for dataset in "${DATASETS[@]}"
do
    echo "Extracting declare for $dataset"
    python preprocess_log.py --log-name $dataset
done

python get_experiments.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do
    for lr in "${LR[@]}"
    do
        for batch_size in "${BATCH_SIZES[@]}"
        do
            for epoch in "${EPOCHS[@]}"
            do
                for n_layers in "${LAYERS[@]}"
                do
                    for input_size in "${INPUT_SIZES[@]}"
                    do
                        for hidden_size in "${HIDDEN_SIZES[@]}"
                            do
                            for template in "${TEMPLATES[@]}"
                            do
                                echo "Training for $dataset with lr=$lr, batch_size=$batch_size, hidden_size=$hidden_size and template=$template"
                                python train.py \
                                --dataset $dataset \
                                --lr $lr \
                                --batch-size $batch_size \
                                --input-size $input_size \
                                --hidden-size $hidden_size \
                                --epochs $epoch \
                                --n-layers $n_layers \
                                --wandb True \
                                --template "$template" \
                                --project-name $PROJECT
                            done
                        done
                    done
                done
            done
        done
    done
done