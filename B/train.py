import numpy as np
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import copy
import random
import wandb
from sklearn.model_selection import StratifiedKFold
from .helper import get_model, get_data_transform, CassavaDataset

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


def train(model_name, num_epochs, train_df, input_size, batch_size, patience, device):

    model = get_model(model_name)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)
    #summary(model, input_size=(batch_size, 3, input_size, input_size))

    n_images = len(train_df)
    n_train = int(n_images * 0.8)  # 60% of the dataset
    n_val = n_images - n_train

    data_transform = get_data_transform(input_size)

    train_df = CassavaDataset(train_df, transform=data_transform)
    train_df, val_df = random_split(
        train_df, 
        [n_train, n_val],
    )

    print('Splitted dataset:')
    print(f'\t- Training set: {len(train_df)}')
    print(f'\t- Validation set: {len(val_df)}')



    train_dl = DataLoader(train_df, batch_size, shuffle=True)
    valid_dl = DataLoader(val_df, batch_size, shuffle=True)

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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
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
                
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
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



def kfold_train(model_name, num_epochs, train_df, num_folds, input_size, batch_size, patience, device):

    #split the 
    sk = StratifiedKFold(n_splits=num_folds, random_state=SEED, shuffle=True)

    model = get_model(model_name, device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_fold = None

    for fold, (train, val) in enumerate(sk.split(train_df, train_df.label)):
        if fold!=4:
            continue
        train_df.loc[val, 'fold'] = fold

        #reset the index after filtering the df based on fold numbers
        train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
        val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

        data_transform = get_data_transform(input_size)
        train_dataset = CassavaDataset(train_data, transform=data_transform)
        val_dataset = CassavaDataset(val_data, transform=data_transform)

        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-4, last_epoch=-1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)


        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs

        current_lr = 0
        counter = 0
        min_valid_loss = np.inf
        
        for epoch in range(num_epochs):
                    
            model.train()
            
            current_lr = optimizer.param_groups[0]['lr']

            for x_batch, y_batch in train_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                        
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
                        
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                            
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


#kfold_train(model_name, num_epochs, train_df, num_folds=5)

#wandb.finish()
