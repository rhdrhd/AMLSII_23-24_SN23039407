#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import json
import torch
from torchvision import models
from torch import nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import copy
from tqdm import tqdm 
from rich.progress import Progress
import timm
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import random
import wandb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

SEED = 23
seed_everything(SEED)



#### DEFINE DATASET
ROOT = 'Datasets'
TRAIN_IMAGES_PATH = os.path.join(ROOT, 'train_images')
train_df = pd.read_csv(os.path.join(ROOT, 'train.csv'))
train_df['path'] = TRAIN_IMAGES_PATH + '/' + train_df['image_id']


class CassavaDataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)  # Reset the index
        #self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        #print("index is " , index)
        if index >= len(self.df):
            raise IndexError('Index out of range')
        img = Image.open(self.df['path'][index])
        #img = np.array(img)
        label = torch.tensor(self.df['label'][index], dtype=torch.long)
        
        if self.transform:
            img = self.transform(img)
        
        return img, label




WIDTH = 512 #224 for ViT
HEIGHT = 512 #224
NUM_CLASSES = 5
BATCH_SIZE = 16
model_name = "resnet50"
num_epochs = 10
patience = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {DEVICE}')




def get_model(name):
    
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        
        in_feat = model.fc.in_features
        
        model.fc = nn.Sequential(
              nn.Linear(in_feat, NUM_CLASSES)
              )
    elif name =="resnext50_32x4d":
        model = timm.create_model('resnext50_32x4d', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        
        in_feat = model.fc.in_features
        
        model.fc = nn.Sequential(
              nn.Linear(in_feat, NUM_CLASSES)
              )
    elif name =="efficientnet_b0":
        model = timm.create_model('efficientnet_b0', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_classes = 5  
        classifier_input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(classifier_input_features, num_classes) 
        )
    elif name =="efficientnet_b3":
        model = timm.create_model('efficientnet_b3', pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_classes = 5  
        classifier_input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(classifier_input_features, num_classes) 
        )
    elif name == "efficientnet_b4":
        model = timm.create_model('tf_efficientnet_b4_ns',pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        num_classes = 5  
        classifier_input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(classifier_input_features, num_classes)
        )
    elif name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        num_classes = 5

        classifier_input_features = model.head.in_features
        model.head = nn.Linear(classifier_input_features, num_classes)
        
    model = model.to(DEVICE)
    
    return model

wandb.init(
    # set the wandb project where this run will be logged
    project="aml2",
    entity= "zirui23",
    # track hyperparameters and run metadata
    config={
    "architecture": model_name,
    "epochs": num_epochs,
    "batch_size": BATCH_SIZE
    }
)

def train(model_name, num_epochs, train_dl, valid_dl):

    model = get_model(model_name)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    summary(model, input_size=(BATCH_SIZE, 3, WIDTH, HEIGHT))

    n_images = len(train_df)
    n_train = int(n_images * 0.8)  # 60% of the dataset
    n_val = n_images - n_train

    train_df = CassavaDataset(train_df, transform=data_transform)
    train_df, val_df = random_split(
        train_df, 
        [n_train, n_val],
    )

    print('Splitted dataset:')
    print(f'\t- Training set: {len(train_df)}')
    print(f'\t- Validation set: {len(val_df)}')



    train_dl = DataLoader(train_df, BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_df, BATCH_SIZE, shuffle=True)

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    current_lr = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    min_valid_loss = np.inf
    
    #with Progress() as progress:
        #training_task = progress.add_task("[red]Training...", total=num_epochs*len(train_dl))
        
    for epoch in range(num_epochs):
        
        model.train()
        
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        for x_batch, y_batch in train_dl:
            
            #progress.update(training_task, advance=1)
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            batch_num += 1
            #if (batch_num % 100 == 0):
                #print(f'Batch number: {batch_num}')
            
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().item()
        
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        

        
        model.eval()
        
        with torch.no_grad():
            
            for x_batch, y_batch in valid_dl:
                
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().item()
                
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        scheduler.step(loss_hist_valid[epoch])

        if accuracy_hist_valid[epoch] > best_acc:
            best_acc = accuracy_hist_valid[epoch]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        print(f'Epoch {epoch+1}:   Train accuracy: {accuracy_hist_train[epoch]:.4f}    Validation accuracy: {accuracy_hist_valid[epoch]:.4f}  Learning Rate: {current_lr}')
        wandb.log({"Epoch": epoch+1, "Train loss": loss_hist_train[epoch], "Validation loss": loss_hist_valid[epoch], "Train accuracy": accuracy_hist_train[epoch], "Validation accuracy":accuracy_hist_valid[epoch], "Learning Rate": current_lr})
    
        if loss_hist_valid[epoch] < min_valid_loss:
            counter = 0
            min_valid_loss = loss_hist_valid[epoch]
        else:
            counter += 1
    
        if counter >= patience:
            break
    
    if model_name in ["resnet18","resnet50"]:
        torch.save(best_model_wts, f'A/{model_name}_best.pth')
    elif model_name in ["efficientnet_b3","efficientnet_b4","vit"]:
        torch.save(best_model_wts, f'B/{model_name}_best.pth')

    model.load_state_dict(best_model_wts)
    
    history = {}
    history['loss_hist_train'] = loss_hist_train
    history['loss_hist_valid'] = loss_hist_valid
    history['accuracy_hist_train'] = accuracy_hist_train
    history['accuracy_hist_valid'] = accuracy_hist_valid
    
    np.savez('history.npz', 
         loss_hist_train=history['loss_hist_train'], 
         loss_hist_valid=history['loss_hist_valid'], 
         accuracy_hist_train=history['accuracy_hist_train'], 
         accuracy_hist_valid=history['accuracy_hist_valid'])
    
    return model, history

data_transform =  transforms.Compose([
    transforms.RandomResizedCrop((WIDTH, HEIGHT)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
])

def kfold_train(model_name, num_epochs, train_df, num_folds=5):

    #split the 
    sk = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    for fold, (train, val) in enumerate(sk.split(train_df, train_df.label)):
        train_df.loc[val, 'fold'] = fold

        #reset the index after filtering the df based on fold numbers
        train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
        val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

        train_dataset = CassavaDataset(train_data, transform=data_transform)
        val_dataset = CassavaDataset(val_data, transform=data_transform)

        train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = get_model(model_name)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)


        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_fold = None
        min_valid_loss = np.inf

        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs
        current_lr = 0
        counter = 0
        
        for epoch in range(num_epochs):
                    
            model.train()
            
            current_lr = optimizer.param_groups[0]['lr']

            for x_batch, y_batch in train_dl:
                        
                #progress.update(training_task, advance=1)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                        
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                        
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.sum().item()
                    
                    
            loss_hist_train[epoch] /= len(train_dl.dataset)
            accuracy_hist_train[epoch] /= len(train_dl.dataset)
                    
            model.eval()
                    
            with torch.no_grad():
                    
                for x_batch, y_batch in valid_dl:
                        
                    x_batch = x_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                            
                    pred = model(x_batch)
                    loss = loss_fn(pred, y_batch)
                    loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                    is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                    accuracy_hist_valid[epoch] += is_correct.sum().item()
                            
            loss_hist_valid[epoch] /= len(valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
                    
            scheduler.step(loss_hist_valid[epoch])

            if accuracy_hist_valid[epoch] > best_acc:
                best_acc = accuracy_hist_valid[epoch]
                best_fold = fold
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'New best accuracy: {best_acc} at fold {best_fold} epoch {epoch}' )
                    
            print(f'Fold {fold}    Epoch {epoch}   Train accuracy: {accuracy_hist_train[epoch]:.4f}    Validation accuracy: {accuracy_hist_valid[epoch]:.4f}  Learning Rate: {current_lr}')
            wandb.log({"Fold": fold, "Epoch": epoch, "Train loss": loss_hist_train[epoch], "Validation loss": loss_hist_valid[epoch], "Train accuracy": accuracy_hist_train[epoch], "Validation accuracy":accuracy_hist_valid[epoch], "Learning Rate": current_lr})

            if loss_hist_valid[epoch] < min_valid_loss:
                min_valid_loss = loss_hist_valid[epoch]
                counter = 0
            else:
                counter += 1
        
            if counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement.')
                break
        
    print(f'Best model was from fold {best_fold} with an accuracy of {best_acc}')

    if model_name in ["resnet18","resnet50","resnext50_32x4d"]:
        torch.save(best_model_wts, f'A/{model_name}_best.pth')
    elif model_name in ["efficientnet_b3","efficientnet_b4","vit"]:
        torch.save(best_model_wts, f'B/{model_name}_best.pth')



kfold_train(model_name, num_epochs, train_df, num_folds=5)



#label_list = []
#prediction_list = []

#with torch.no_grad():
#    for image, label in tqdm(test_dl):
#        
#        image = image.to(DEVICE)
#        logits = best_model(image)
#        probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()
#        prediction = np.argmax(probs, axis=1)
#        label_list += label.numpy().tolist()
#        prediction_list += prediction.tolist()


# Assuming 'history' is a dictionary that contains loss values for training and validation
# And also assuming that the number of epochs is contained in the length of the loss lists

#epochs = list(range(1, len(hist['loss_hist_train']) + 1))

#plt.figure(figsize=(10, 5))
#plt.plot(epochs, hist['loss_hist_train'], label='Training Loss')
#plt.plot(epochs, hist['loss_hist_valid'], label='Validation Loss')

#plt.title('Training and Validation Loss per Epoch')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.grid()

#if model_name in ["resnet50"]:
#    plt.savefig(f"A/{model_name}_loss.png")
#if model_name in ["efficientnet_b3","efficientnet_b4", "vit"]:
#    plt.savefig(f"B/{model_name}_loss.png")


#print(classification_report(label_list, prediction_list))


#cm = confusion_matrix(label_list, prediction_list)
#plt.figure(figsize=(6, 6))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, linewidth=1, linecolor='white')
#plt.xlabel('Predicted labels')
#plt.ylabel('True labels')
#plt.title('Confusion Matrix')
#if model_name in ["resnet50"]:
#    plt.savefig(f"A/{model_name}_confusion_m.png")
#if model_name in ["efficientnet_b3", "efficientnet_b4","vit"]:
#    plt.savefig(f"B/{model_name}_confusion_m.png")

wandb.finish()
