# ELEC0135_Project_SN23039407

This repo presents the development and evaluation of two specialized machine learning models for Cassava Leaf Disease Classification tasks.

**main.py**: Contains the run file of the project, including the training and testing options of the models for Task A and Task B.

**A**: Contains Jupyter Notebooks version code is stored in this folder

**B**: Contains the core of the project train.py, which executes the training

## How to start
1. Create a new conda environment from environment.yml file.
```
conda env create -f environment.yml
```
2. Activate this conda virtual environment. 
```
conda activate aml2
```
3. Download kaggle dataset from Cassava Leaf Disease Classification [link to competition](https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data), unzip the file and place them into Datasets folder

4. Run main.py if all the dependencies required for the current project are already installed. 

```
python main.py
```
### 5 Fold (10 Epoch) (Batch Size: 16)

| Model                 | Weights  | Dropout | Accuracy (%) | Image Size | Seed |
|-----------------------|----------|---------|--------------|------------|------|
| ResNet50              | Frozen   | No      | 80.09        | 512        | 23   |
| ResNeXt50_32x4d       | Frozen   | No      | 75.93        | 512        | 23   |
| EfficientNet-B3       | Frozen   | No      | 80.74        | 512        | 23   |
| EfficientNet-B4       | Unfrozen | No      | -            | 384        | 23   |
| ViT-Large Patch32_384 | Unfrozen | No      | -            | 384        | 729  |

### 5 Fold (15 Epoch) (Batch Size: 16)

| Model            | Weights  | Dropout | Image Size | Seed |
|------------------|----------|---------|------------|------|
| EfficientNet-B4  | Unfrozen | 0.3     | 384        | 23   |

### Random Split (20 Epoch)

| Model            | Weights | Dropout | Image Size | Seed | Notes            |
|------------------|---------|---------|------------|------|------------------|
| EfficientNet-B4  | Frozen  | No      | 512        | 23   | -                |
| EfficientNet-B4  | Frozen  | No      | 512        | 23   | Duplicate entry  |
