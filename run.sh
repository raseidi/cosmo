#cd #/bin/bash
DIR="/home/seidi/datasets/logs"
CONDITION=("trace_time" "resource_usage")
DEVICE="cuda"

find $DIR -print0 | while IFS= read -r -d '' directory
do 
    if [[ $directory == *train_test ]] 
    then
        dataset=$(echo $directory | rev | cut -d "/" -f2 | rev)
        echo "Running $dataset"
        if [[ $dataset != "bpi15" && $dataset != "PrepaidTravelCost" ]]
        then
            # python3.8 prepare_data.py --path $directory --dataset $dataset
            for CO in ${CONDITION[@]}
            do
                python3.8 train.py --dataset $dataset --condition $CO --device $DEVICE
            done
        fi        
    fi
done


