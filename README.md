# CoSMo: a Framework for Implementing Conditioned Process Simulation Models

This is a Pytorch implementation of the [CoSMo paper](.):

```
@Article{Oyamada23cosmo,
  author  = {Rafael Oyamada and Gabriel Tavares and Paolo Ceravolo},
  title   = {CoSMo: a Framework for Implementing Conditioned Process Simulation Models},
  journal = {arXiv preprint},
  year    = {2023},
}
```

## Enviroment

To install and activate the enviroment, run:

```
conda env create --file env.yml
conda activate cosmo
```

The current version uses wandb for tracking experiments, it is necessary to [set up your account](https://docs.wandb.ai/guides/app/settings-page/user-settings).


## Preparation

We use [these benchmarked event logs](https://github.com/hansweytjens/predictive-process-monitoring-benchmarks/). Follow the instructions from the link to prepare a specific dataset.

## CoSMo Pipeline

After installing the enviroment and setting up the benchmarked event logs, the experiments can be reproduced by running the following scripts, respectively:

```
python3 prepare_data.py
python3 train.py
python3 simulation.py
python3 evaluation.py
```

Each script and its respetive arguments are described below.

### Prepare data

Default arguments: 

```
prepare_data.py \
--path benchmarked/bpi20/RequestForPayment/train_test/ \
--dataset RequestForPayment \
[--overwrite]
``` 

In this script, `--path` denotes the output directory from the benchmark repository containing the files `train.csv` and `test.csv`. The arugment `--dataset` is an auxiliary parameter in order to distinguish, e.g., the BPI20 variations. Finally, `--overwrite` is an optional argument that replaces the output of `prepare_data.py` if any exists.

This script essencially labels cases according to the conditioning function defined in the paper. This repository can be easily extended by implementing custom conditions.

### Train

With preparing the datasets, the training with custom hyperparameters can be performed using this script. Default arguments:

```
python3 train.py \
--dataset RequestForPayment \
--condition resource_usage \
--device cuda \
--lr 0.0001 \
--epochs 50 \
--optimizer adam \
--weight-decay 0 \
--project-name bpm23
```

The `--condition` regards the function name for conditioning (labeling) cases. We suggest running `RequestForPayment` first for testing since it is the lighest event log (using GPU, 50 epochs finish in a few minutes).

***Note***: this script can be skipped for reproducibility, since the `simulation.py` opens a model or retrains it using the same hyperparameters from the paper. ToDo: list best hyperparameters for each dataset here.

### Simulation

In this current version of the repository, this script runs the simulations for all datasets. In future versions we will include arguments to make it easier. The simulated datasets are persisted in `results/simulations/<dataset>_<condition><architecture>_on_going.csv`. Just run:

```
python3 simulation.py
```

### Evaluation

This scrip shares the same limitation than previous one. Results are saved in `results/`. Just run:

```
python3 evaluation.py
```