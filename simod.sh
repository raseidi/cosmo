# /bin/bash
# this script runs simod for the datasets in the `DATASETS` array and saves the runtime in the `runtime.csv` file

# function to run simod
function run_simod {

    # get the parameters
    dataset=$1
    rm -rf simod/output/$dataset/
    mkdir -p simod/output/$dataset/
    
    # if you do not have simod installed, refer to
    # https://github.com/AutomatedProcessImprovement/Simod
    # notice: in our cosmo repository, we install simod in a different enviroment
    eval "$(conda shell.bash hook)"
    conda activate simod

    # check simod
    simod --version

    SECONDS=0 # internal variable
    simod --configuration simod/config.yml --output simod/output/$dataset/
    # simod --configuration default.yml # original from simod
    elapsed_time=$SECONDS

    # simod actually has exit code 0 even if it fails
    echo "$dataset,$elapsed_time,$?" >> simod/runtime.csv
    # if [ $? -eq 0 ]; then
    #     echo "$dataset,$elapsed_time,$?" >> simod/runtime.csv
    # else
    #     echo "$dataset,$elapsed_time,$?"
    # fi
}

function prepare_simod {
    # get the parameters
    dataset=$1

    eval "$(conda shell.bash hook)"
    conda activate cosmo-ltl

    python simod/prepare_simod.py --log $dataset
}


# check if `runtime.csv` exists
if [ ! -f simod/runtime.csv ]; then
    # create the file with the header
    echo "dataset,runtime,status" > simod/runtime.csv
fi

declare -a DATASETS=("sepsis" "bpi12" "bpi13_incidents" "bpi13_problems" "bpi17" "bpi19" "bpi20_permit" "bpi20_prepaid" "bpi20_req4pay")

# iterate over the datasets
for dataset in "${DATASETS[@]}"
do
    prepare_simod $dataset
    run_simod $dataset
done