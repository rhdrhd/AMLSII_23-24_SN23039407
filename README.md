# ELEC0135_Project_SN23039407

This repo presents the development and evaluation of two specialized machine learning models for Cassava Leaf Disease Classification tasks.

**main.py**: Contains the run file of the project, including the training and testing options of the models for Task A and Task B.

**A**: Contains the test files used in development and Jupyter Notebooks version code is stored in this folder

**B**: Contains the core of the project, including train.py, helper.py and config.json, which together execute the training process. Model configurations are stored in config.json.

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
5. The training configuration can be changed in B/config.json. The current configuration is set to train the EfficientNet_B4 model with unfrozen weights and one dropout of 0.3 for 30 epochs, with batch size as 16, early stopping patience as 7 epochs, initial learning rate as 0.001, image input size as 384, and only the best fold (5th) is used for training.

### Model Log

| Model                       | Weights  | Dropout   | Image Size | Seed | Epochs | Accuracy (%) |
|-----------------------------|----------|-----------|------------|------|--------|--------------|
| ResNet50                    | Frozen   | No        | 512        | 23   | 10     | 80.09        |
| EfficientNet_B4             | Frozen   | No        | 384        | 23   | 10     | 80.74        |
| EfficientNet_B4             | Unfrozen | No        | 384        | 23   | 10     | 86.80        |
| EfficientNet_B4             | Unfrozen | Yes (0.3) | 384        | 23   | 15     | 86.07        |
| EfficientNet_B4             | Unfrozen | Yes (0.3) | 384        | 23   | 30     | 87.73        |
| ViT-Large_Patch32_384       | Unfrozen | No        | 384        | 729  | 10     | 65.34        |
