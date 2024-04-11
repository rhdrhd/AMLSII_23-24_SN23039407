import json
import torch
import os
import pandas as pd
import wandb
import B.train as train


# Load the configuration
with open('B/config.json', 'r') as f:
    config = json.load(f)

# Accessing configuration details
model_name = config['model']['name']
input_size = config['model']['input_size']
batch_size = config['training']['batch_size']
num_epochs = config['training']['epochs']
patience = config['training']['patience']
num_folds = config['training']['num_folds']

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device} is used for training')

# Load dataset
ROOT = 'Datasets'
TRAIN_IMAGES_PATH = os.path.join(ROOT, 'train_images')
train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
train_df['path'] = TRAIN_IMAGES_PATH + '/' + train_df['image_id']


# Track the progress using wandb
wandb.init(
    project="aml2",
    entity= "zirui23",
    # track hyperparameters and run metadata
    config={
    "architecture": model_name,
    "epochs": num_epochs,
    "batch_size": batch_size,
    "input_size": input_size
    }
)

train.kfold_train(model_name, num_epochs, train_df, num_folds, input_size, batch_size, patience, device)

wandb.finish()