#cd #/bin/bash
DIR="/home/seidi/datasets/logs"
CONDITION=("resource_usage")
PROJECT_NAME="bpm23"
DEVICE="cuda"

find $DIR -print0 | while IFS= read -r -d '' directory
do 
    if [[ $directory == *train_test ]] 
    then
        dataset=$(echo $directory | rev | cut -d "/" -f2 | rev)
        echo "Running $dataset"
        if [[ $dataset != "bpi15" ]]
        then
            # python3.8 prepare_data.py --path $directory --dataset $dataset
            for CO in ${CONDITION[@]}
            do
                python3.8 train.py --dataset $dataset --condition $CO --device $DEVICE --project-name $PROJECT_NAME
            done
        fi        
    fi
done


