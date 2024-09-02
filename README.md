# COnditioned process Simulation MOdels (CoSMo)

Codebase for the [CoSMo paper](https://link.springer.com/chapter/10.1007/978-3-031-70396-6_19) accepted at the BPM'24 conference. 

---

## Table of Contents

- [COnditioned process Simulation MOdels (CoSMo)](#conditioned-process-simulation-models-cosmo)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
    - [Datasets](#datasets)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Discovering declare rules](#discovering-declare-rules)
    - [Training](#training)
    - [Simulating](#simulating)
  - [I/O files](#io-files)
    - [Event logs](#event-logs)
    - [Trained models](#trained-models)
    - [Simulated logs](#simulated-logs)
  - [Citation](#citation)

## Overview


In process mining, the most popular process simulation models are fully data-driven, as traditional learning-based solutions do not offer sufficient flexibility for real-world applications. This project aims to develop a constrained learning approach, enabling deep neural networks to meet user-defined constraints at simulation time. This approach addresses the flexibility gap in such existing solutions.

I have made significant efforts to ensure this repository is reproducible and understandable. If you face any issues, please open an issue or [contact me](https://raseidi.github.io/pages/contact.html) via email or social media.


### Datasets

**NOTE**: these statistics are after the preprocessing step (cached logs), thus the number of activities and events might differ from the original ones from the [4tu repository](https://data.4tu.nl/).

<center>

| Event log 	| # activities 	| # events (10^3) 	| # cases (103) 	| Split 	|
|:---------:	|:------------:	|:---------------:	|:-------------:	|:-----:	|
|   BPI12   	|      23      	|      104.82     	|  11.05 ± 9.64 	|  Unbiased¹|
|   BPI13   	|       6      	|       4.89      	|  3.69 ± 4.09  	|  Unbiased¹|
|   BPI17   	|      26      	|     1210.81     	| 38.44 ± 17.96 	|  Unbiased¹|
|   BPI20   	|      50      	|      82.22      	|  12.01 ± 5.46 	|  Unbiased¹|
|   SEPSIS  	|       8      	|       9.87      	|  10.37 ± 3.9  	|  pm4py²  	|
</center>

¹ [Creating Unbiased Public Benchmark Datasets with Data Leakage Prevention for Predictive Process Monitoring](https://arxiv.org/abs/2107.01905)

² [pm4py library](https://pm4py.fit.fraunhofer.de/static/assets/api/2.7.11/generated/pm4py.ml.split_train_test.html)


## Installation

Instructions for setting up the environment and installing the dependencies for this repository. Make sure to use python3.10.

**Note**: Tested only on Ubuntu.


```bash
# Clone this repository
git clone https://github.com/raseidi/cosmo.git

# Navigate to the project directory
cd cosmo

# Create a virtual environment (tested with conda 23.11.0 only)
conda create --name cosmo python=3.10

# using pyenv should be
python3.10 -m venv cosmoenv
source cosmoenv/bin/activate  # On Windows use `cosmoenv\Scripts\activate`

# Install required packages
pip install -r requirements.txt

```

## Usage

Download all the [cached data](https://drive.google.com/file/d/1M6AdQF6Ui0ypxU_0X-1Z9hWUljm7PJuH/view?usp=sharing). For debugging purposes, we suggest using the `sepsis` event log, as its preprocessing and training times are relatively fast, especially if a GPU is available.

Extract the cached data and make sure the repository looks like:

```markdown
cosmo/
data/
    ├── bpi12/
    ├── bpi13_problems/
    ├── bpi17/
    ├── bpi20_permit/
    ├── sepsis/
    └── simulation/
other files...
```

In general, to reproduce the results of this repository, follow these steps:
1. Extract the declare rules from event logs.
2. Train the model(s).
3. Simulate processes based on the trained model.

### Discovering declare rules

You can either extract rules for a single dataset:

```bash
python preprocess_log.py --log-name sepsis
```

or run the following script to extract rules from all event logs:

```bash
chmod +x extract_declare.sh
./extract_declare.sh
```

### Training

You can either train one single instance with a custom configuration

```bash
python train.py \
    --dataset sepsis \
    --template choice \
    --backbone crnn \
    --lr 0.0005 \
    --batch-size 64 \
    --hidden-size 256 \
    --input-size 32 \
    --n-layers 1 \
    --epochs 50
```

or reproduce the whole paper running the following bash script (it might take a few hours)

```bash
chmod +x reproduce_paper.sh

./reproduce_paper.sh
```

If you want to optimize different hyperaparameters, edit the `train.sh` script and run it as the above script. Unfortunately, though, the current simulation scripts support only the default models trained using the `reproduce_paper.sh` script.

Find below the available arguments for the `train.py` script:

| Argument         	| Type    	| Default Value     	| Choices                                                           	| Decription                                                       	|
|------------------	|---------	|-------------------	|-------------------------------------------------------------------	|------------------------------------------------------------------	|
| `--dataset`      	| `str`   	| sepsis            	| `["sepsis", "bpi12", "bpi13_problems",  "bpi17", "bpi20_permit"]` 	| Event log to be used                                             	|
| `--lr`           	| `float` 	| 5e-4              	|                                                                   	| Learning rate                                                    	|
| `--batch-size`   	| `int`   	| 32                	|                                                                   	| Batch size                                                       	|
| `--weight-decay` 	| `float` 	| 1e-5              	|                                                                   	| Weight decay for training regularization                         	|
| `--epochs`       	| `int`   	| 100               	|                                                                   	| Number of epochs                                                 	|
| `--device`       	| `str`   	| cuda              	| `["cuda", "cpu"]`                                                 	| Whether it should run on gpu or cpu                              	|
| `--hidden-size`  	| `int`   	| 32                	|                                                                   	| Number of hidden units                                           	|
| `--input-size`   	| `int`   	| 8                 	|                                                                   	| Embedding size (input for the hidden layer)                      	|
| `--project-name` 	| `str`   	| `"cosmo-bpm-sim"` 	|                                                                   	| Project name, only if `wandb` is enabled                         	|
| `--n-layers`     	| `int`   	| 1                 	|                                                                   	| Number of (constrained) recurrent layers                         	|
| `--wandb`        	| `str`   	| False             	| True if passed, False otherwise                                   	| Enable or disable of wandb                                       	|
| `--template`     	| `str`   	| `"choice"`        	| `["existence", "choice", "positive relations", "all"]`            	| Declare tempalte to be trained                                   	|
| `--backbone`     	| `str`   	| `"crnn"`          	| `["crnn", "vanilla"]`                                             	| Backbones (constrained rnn proposed in this work or vanilla rnn) 	|

**NOTE**: `wandb` is not included in the requirements but `train.py` script accepts the flag `--wandb` to enable it, if you decide to install it.

### Simulating

As specified in the previous table, you can use the constrained or vanilla rnn for training. Accordingly, all the simulations can be reproduced by running the following script after training

```bash
chmod +x simulation_<backbone>.sh
./simulation_<backbone>.sh
```

where `backbone` is either `crnn` or `vanilla`.

Currently, running a simulation for a custom model trained with different hyperparameters is not supported. That requires a bit of refactoring on the codebase (sorry about that).

## I/O files

### Event logs

How the `data/` directory is structured:

```markdown
# Example structure
data/
    └── bpi12/
        ├── cached_train_test/      # datasets with its respective declare rules
            ├── dataset_choice_test.pt
            └── dataset_choice_train.pt        
        ├── declare/                # declare rules extracted from bpi12
            └── constraints.pkl     
        ├── train_test/              # raw event logs from the unbiased split paper
            ├── train.csv  
            └── test.csv        
        ├── cached_log.pkl          # event log preprocessed
        └── log.xes                 # original log downloaded from the 4tu repository
```

### Trained models

The models are persisted in the `models/` directory. Here is an example of how it looks like:

```markdown
models/
    └── sepsis/
        └── backbone=crnn-templates=choice-lr=0.0005-bs=64-hidden=256-input=32.pth
```

### Simulated logs

The simulated logs will be persisted under the `data/simulation/` directory, which is organized by the two subfolders `crnn` and `vanilla`. Example:

```markdown
data/simulation/
    └── crnn/
        └── dataset=sepsis-template=positive relations-sim_strat=original-sampling_strat=multinomial.pkl
    └── vanilla/
        └── dataset=sepsis-template=all-sim_strat=original.pkl
```

## Citation

```bibtex
@InProceedings{Oyamada2023cosmo,
  author="Oyamada, Rafael Seidi
  and Marques Tavares, Gabriel
  and Barbon Junior, Sylvio
  and Ceravolo, Paolo",
  editor="Marrella, Andrea
  and Resinas, Manuel
  and Jans, Mieke
  and Rosemann, Michael",
  title="CoSMo: A Framework to Instantiate Conditioned Process Simulation Models",
  booktitle="Business Process Management",
  year="2024",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="328--344",
  isbn="978-3-031-70396-6"
}
```
