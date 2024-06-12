# bin/bash

# best configs for each dataset found in the hyperparameter search
# not optmal, but good enough to reproduce the results

# === sepsis all vanilla ===
python train.py --dataset sepsis --template all --backbone vanilla --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === sepsis "positive relations" crnn ===
python train.py --dataset sepsis --template "positive relations" --backbone crnn --lr 0.0005 --batch-size 16 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === sepsis existence crnn ===
python train.py --dataset sepsis --template existence --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === sepsis choice crnn ===
python train.py --dataset sepsis --template choice --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi20_permit all vanilla ===
python train.py --dataset bpi20_permit --template all --backbone vanilla --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi20_permit "positive relations" crnn ===
python train.py --dataset bpi20_permit --template "positive relations" --backbone crnn --lr 5e-05 --batch-size 16 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi20_permit existence crnn ===
python train.py --dataset bpi20_permit --template existence --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi20_permit choice crnn ===
python train.py --dataset bpi20_permit --template choice --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi17 all vanilla ===
python train.py --dataset bpi17 --template all --backbone vanilla --lr 0.0005 --batch-size 64 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi17 "positive relations" crnn ===
python train.py --dataset bpi17 --template "positive relations" --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi17 existence crnn ===
python train.py --dataset bpi17 --template existence --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi17 choice crnn ===
python train.py --dataset bpi17 --template choice --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi13_problems all vanilla ===
python train.py --dataset bpi13_problems --template all --backbone vanilla --lr 5e-05 --batch-size 16 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi13_problems "positive relations" crnn ===
python train.py --dataset bpi13_problems --template "positive relations" --backbone crnn --lr 5e-05 --batch-size 16 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi13_problems existence crnn ===
python train.py --dataset bpi13_problems --template existence --backbone crnn --lr 5e-05 --batch-size 16 --hidden-size 256 --input-size 32 --n-layers 1 --epochs 50

# === bpi13_problems choice crnn ===
python train.py --dataset bpi13_problems --template choice --backbone crnn --lr 5e-05 --batch-size 16 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi12 all vanilla ===
python train.py --dataset bpi12 --template all --backbone vanilla --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi12 "positive relations" crnn ===
python train.py --dataset bpi12 --template "positive relations" --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi12 existence crnn ===
python train.py --dataset bpi12 --template existence --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

# === bpi12 choice crnn ===
python train.py --dataset bpi12 --template choice --backbone crnn --lr 0.0005 --batch-size 64 --hidden-size 128 --input-size 32 --n-layers 1 --epochs 50

