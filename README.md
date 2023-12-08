# CoSMo: a Framework for Implementing Conditioned Process Simulation Models

This is a Pytorch implementation of the [CoSMo preprint](https://arxiv.org/abs/2303.17879):

```
@Article{Oyamada23cosmo,
  author  = {Rafael Oyamada and Gabriel Tavares and Paolo Ceravolo},
  title   = {CoSMo: a Framework for Implementing Conditioned Process Simulation Models},
  journal = {arXiv abs/2303.17879},
  year    = {2023},
}
```

## Enviroment

The following commands create an enviroment and install the dependencies.

```bash
conda create -n python=3.10 cosmo pip
conda activate cosmo
pip install -r requirements.txt
```

We optionally use wandb for tracking experiments, it is necessary to [set up your account though](https://docs.wandb.ai/guides/app/settings-page/user-settings).

---

## Download event logs 

TODO: 

- upload to cloud before including them here
- set optional wandb

## Preprocessing and training

1. Run `preprocess_log.py` to preprocess the event log and extract the DECLARE rules. Arguments:
   1. `--log-name`: event log to be preprocessed
2. Subsequently, run `train.py` to train a model. Arguments:
   - `--dataset` (str): event log name
   - `--template` (str): declare template
   - `--lr` (float): learning rate
   - `--batch-size` (int): batch size
   - `--weight-decay` (float): weight decay rate
   - `--epochs` (int): number of epochs
   - `--device` (str): torch device (cpu or cuda)
   - `--hidden-size` (int): recurrent layer hidden size 
   - `--input-size` (int): recurrent layer input size 
   - `--n-layers` (int): number of recurrent layers
   - `--wandb` (bool): if wandb should be used or not
   - `--project-name` (str): wandb project

Usage example for preprocessing:

```bash
preprocess_log.py \
--log--name sepsis 
```

And for training:

``` bash
python train.py \
  --dataset sepsis \
  --template existence \
  --lr 0.0005 \
  --batch-size 64 \
  --epochs 50 \
  --device cuda \
  --hidden-size 256 \
  --input-size 128 \
  --n-layers 1 \
  --wandb True \
  --project-name cosmo 
```

The trained model is outputed in the directory `models/<event-log>/`.

---

## Simulation

After training the model, run `./simulate.py`. Arguments:

- `--log-name`: event log to be used as input


The simulated log is ouputed at `data/simulation/`.

