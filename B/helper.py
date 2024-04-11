import torch
from torchvision import models
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import timm
import json


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

def get_model(name, DEVICE):
    
    with open('B/config.json', 'r') as f:
        config = json.load(f)

    NUM_CLASSES = config['model']['num_classes']

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
        model = timm.create_model('tf_efficientnet_b4.ns_jft_in1k',pretrained=True)

        num_classes = 5  
        classifier_input_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(classifier_input_features, num_classes, bias=True)
        )

    elif name == "vit":
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        num_classes = 5

        classifier_input_features = model.head.in_features
        model.head = nn.Linear(classifier_input_features, num_classes)
        
    model = model.to(DEVICE)
    
    return model

def get_data_transform(input_size):
    data_transform =  transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
    ])
    return data_transform