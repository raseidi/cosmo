#bin/bash

LR=(0.00005)
BATCH_SIZES=(8 32)
EPOCHS=(2)
LAYERS=(1)
INPUT_SIZES=(32)
HIDDEN_SIZES=(128)
RANK_DIMS=(8 64)
LORA_ALPHA=(8 32 128)
WANDB=True
declare -a LORA=(False True)
declare -a BACKBONES=("gpt2")

PROJECT="cosmo-bpm24"

# declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi17" "bpi19" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")
declare -a DATASETS=("bpi17" "bpi19")


# for dataset in "${DATASETS[@]}"
# do
#     echo "Extracting declare for $dataset"
#     python preprocess_log.py --log-name $dataset
# done

python get_experiments.py --project $PROJECT

for dataset in "${DATASETS[@]}"
do
    for backbone in "${BACKBONES[@]}"
    do
        for lr in "${LR[@]}"
        do
            for batch_size in "${BATCH_SIZES[@]}"
            do
                for lora in "${LORA[@]}"
                do
                    if [ "$lora" = "False" ]; then
                        echo "Training $backbone for $dataset lora=$lora"
                        python train.py \
                        --dataset $dataset \
                        --lr $lr \
                        --batch-size $batch_size \
                        --backbone $backbone \
                        --project-name $PROJECT \
                        --lora $lora \
                        --wandb $WANDB
                    else
                        for rank_dim in "${RANK_DIMS[@]}"
                        do
                            for lora_alpha in "${LORA_ALPHA[@]}"
                            do
                                echo "Training $backbone for $dataset lora=$lora"
                                python train.py \
                                --dataset $dataset \
                                --lr $lr \
                                --batch-size $batch_size \
                                --backbone $backbone \
                                --r-rank $rank_dim \
                                --lora-alpha $lora_alpha \
                                --project-name $PROJECT \
                                --lora $lora \
                                --wandb $WANDB
                            done
                        done
                    fi
                done
            done
        done
    done
done