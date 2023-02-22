#cd #/bin/bash
DIR="/home/seidi/datasets/logs"

BATCH_SIZE=(64)
EPOCHS=(50)
LEARNING_RATE=(0.00005)
WEIGHT_DECAY=(0.0)
LR_SCHEDULER=true
CONDITION="trace_time"
DEVICE="cuda"

find $DIR -print0 | while IFS= read -r -d '' directory
do 
    if [[ $directory == *train_test ]] 
    then 
        dataset=$(echo $directory | rev | cut -d "/" -f2 | rev)
        echo "Running $dataset"
        python3.8 prepare_data.py --path $directory --dataset $dataset --condition $CONDITION
            
        for BS in ${BATCH_SIZE[@]}
        do
            for EP in ${EPOCHS[@]}
            do
                for LR in ${LEARNING_RATE[@]}
                do
                    for WD in ${WEIGHT_DECAY[@]} 
                    do
                        python3.8 train.py -b $BS --dataset $dataset --epochs $EP --lr $LR --wd $WD --condition $CONDITION --device $DEVICE
                    done
                done
            done
        done
    fi
done


