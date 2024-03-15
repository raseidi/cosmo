#bin/bash

LR=(0.0005)
BATCH_SIZES=(64)
EPOCHS=(25)
LAYERS=(1)
INPUT_SIZES=(32)
HIDDEN_SIZES=(128)
declare -a BACKBONES=("vanilla")
declare -a TEMPLATES=("existence" "choice" "positive relations")

PROJECT="cosmo-bpm-sim"

# declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi17" "bpi19" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")
declare -a DATASETS=("bpi17")

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
                                # if backbone is crnn, iterate over the templates
                                # otherwise, just train the vanilla model
                                if [ $backbone == "vanilla" ]
                                then
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
                                    --backbone $backbone \
                                    --project-name $PROJECT
                                else
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
                                        --backbone $backbone \
                                        --project-name $PROJECT
                                    done
                                fi
                            done
                        done
                    done
                done
            done
        done
    done
done