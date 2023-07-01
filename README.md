# mmocr-pipeline

## Description

This is an API to assist in processing and creating configurations and JSON files needed to
train and test on the MMOCR library. As part of this README file, I have included general
environment setup for running this pipeline as well as the setup for running MMOCR.

## Features
- All you need is a single file for all tasks
- Generate Configuration and JSON files for multiple datasets
- Generate basic training/testing configuration file for available detection and recognition frameworks in MMOCR
- Includes a Box Transformer class to transform between different types of bounding boxes (currently supports types: VOC, ICDAR, QUAD, COCO)
- Includes a custom generator to generate bounding boxes given text and image using MMOCR (NOT FINISHED YET)
- Can include custom dataset by defining a class for each under dataset_lib folder (all you need to define is the process function, by inheriting from MMOCRDataset)



## Environment Configuration

### Create Conda Environment

Preferred Python version 3.8, limit 3.10
```commandline
>>> cd <<Directory of Choice>>
>>> conda create --name ocrenv
>>> conda activate ocrenv
>>> conda install python=3.8 conda pip 
```

### Install Dependencies

Install Pytorch (CUDA: 11.7 is preferred) following Link (https://pytorch.org/get-started/locally/)
```commandline
>>> conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Setup MMOCR
```commandline
>>> pip install -U openmim
>>> mim install mmengine
>>> mim install ‘mmcv>=2.0.rc1’
>>> mim install ‘mmdet>=3.0.0rc0’ 
>>> git clone https://github.com/open-mmlab/mmocr.git 
>>> cd mmocr 
>>> pip install -r requirements.txt 
>> pip install -v -e .
```

Others (Make sure to check if the following has already been installed alongside the former)
```commandline
>>> conda install pillow
>>> conda install pandas
>>> conda install numpy
>>> conda install -c conda-forge datasets 
>>> pip install levenshtein
>>> pip install pyarrow
>>> conda install json
>>> conda install argparse
```

## Pipeline

### Setting up the single config file

Three important defenition: dataset_dict, det_model_dict and recog_model_dict

For No task (Empty configuration)
```python
dataset_dict = None

det_model_dict = None

recog_model_dict = None
```

### Dataset dict
When dataset_dict is None, no dataset's would be processed

You can either provide a list of datasets to process under dataset_dict or a single dataset

Each dataset would be defined using the following setting

```python
dataset_dict = [
    dict(
        type="cordv2",   #Available types include "cordv2", "ing", "meme", "glosat"
        init_params= dict(name="cordv2", tasks = ["det"], save_dir=None, generator=None),
        prepare_params=dict(train=dict(img_paths=None, ann_paths=None, split="train"),
                            test=dict(img_paths=None, ann_paths=None, split="test"),
                            val=dict(img_paths=None, ann_paths=None, split="val"))
    ),
]
```

The above version is a list, you can also provide just the dictionary if you want to process only that dataset

The init_params and prepare_params are common for all datasets and should not be changed

If changed in custom definition, do remember to include the parameter where needed

Currently we support the following datasets:
- CORD dataset: "cordv2"
- Ingredient dataset (custom): "ing"
- Meme dataset (custom): "meme"
- Glosat dataset: "glosat"

Supports tasks: "det" and "recog"

### Defining a custom generator under init_params:

Define the detection, recognition, custom weights, device and save directory for 
ocr-based bounding box generator.

max_neighbours (must be a odd number) defines the window size for considering neighbouring predicted text

Recommends 5 for max_neighbours

Given max_neighbours look (k-1)/2 to the right and left
```python
generator = dict(
    det_model="dbnet", det_weights=None, rec_model="satrn", rec_weights=None,
    device="cuda:0", save_dir="./box_gen/", max_neighbours=5
)
```

### Model dict

For models, we support detection and recognition configuration, where the definition for both are similar.
Like dataset_dict, you can also provide a list of dicts or a single one if needed

### Det Model Dict

Available schedules, backbones and necks for each model can be found under "./mmocr_config_writers/configs/det_configs.py"

```python
det_model_dict = dict(
    train_datasets=["cordv2.py"], #Single or Multiple training datasets
    val_datasets=["cordv2.py"], #Single, Multiple or None validation datasets
    test_datasets=["cordv2.py"], #Single or Multiple testing datasets
    model="dbnet", #Model name depending on task
    backbone="resnet18", #Backbone for model
    neck="fpnc", #Neck of model
    base=None, #Base not needed to be defined for detection model
    epochs=40, #Maximum epoch
    schedule=None, #Predefined Learning rate schedulers, optimizers
    has_val=True, # If validation is included
    train_batch_size=16, # Batch size for training
    test_batch_size=1, # Batch size for testing and validation
    contents = dict( #Dict defining the hyperparameters
        log_interval=1, # Log at each k epoch
        checkpoint_interval=1, # Save at each k epoch
        optimizer_params=dict( #Follows pytorch defenitions for parameters
            type="SGD", lr=0.007, momentum=0.9, weight_decay=0.0001
        ),
        schedulers=[ #List of schedulers, follows pytorch defenitions
            dict(type="ConstantLR", factor=1.0)
        ],
        cfgs=dict( #No need for val/test cfg to be defined as they are pretty 
            # much not need for changing
            #train_cfg, define max_epoch and validation interval (every k epochs, validate model)
            train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=40, val_interval=1),
            val_cfg=None,
            test_cfg=None
        ),
    )
```

### Recog Model Dict

Available schedules, bases for each model can be found under "./mmocr_config_writers/configs/recog_configs.py"

```python
recog_model_dict = dict(
    train_datasets=["cordv2.py"], #Single or Multiple training datasets
    val_datasets=["cordv2.py"], #Single, Multiple or None validation datasets
    test_datasets=["cordv2.py"], #Single or Multiple testing datasets
    model="abinet", #Model name depending on task
    backbone=None, #Backbone not needed to be defined for recognition model
    neck=None, #Neck not needed to be defined for recognition model
    base="_base_abinet-vision.py", #Base for model
    epochs=40, #Maximum epoch
    schedule=None, #Predefined Learning rate schedulers, optimizers
    has_val=True, # If validation is included
    train_batch_size=16, # Batch size for training
    test_batch_size=1, # Batch size for testing and validation
    contents = dict( #Dict defining the hyperparameters
        log_interval=1, # Log at each k epoch
        checkpoint_interval=1, # Save at each k epoch
        optimizer_params=dict( #Follows pytorch defenitions for parameters
            type="SGD", lr=0.007, momentum=0.9, weight_decay=0.0001
        ),
        schedulers=[ #List of schedulers, follows pytorch defenitions
            dict(type="ConstantLR", factor=1.0)
        ],
        cfgs=dict( #No need for val/test cfg to be defined as they are pretty 
            # much not need for changing
            #train_cfg, define max_epoch and validation interval (every k epochs, validate model)
            train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=40, val_interval=1),
            val_cfg=None,
            test_cfg=None
        ),
    )
```

### Running prepare

Once above configuration file is defined run the following code with following optional flags
- '-nd' or '--no-dataset' -> Do not process defined datasets
- '-ndm' or '--no-det-model' -> Do not process defined detection model
- '-nrm' or '--no-recog-model' -> Do not process defined recognition model

```commandline
>>> python prepare.py <<config_name>>.py
```

Example, if custom config file is named prepare_config_2.py
```commandline
>>> python prepare.py prepare_config_2.py
```

