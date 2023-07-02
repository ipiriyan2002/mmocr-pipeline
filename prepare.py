import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config

#Custom Imports
from dataset_lib import dataset_ids as ids
from mmocr_config_writers.model_config import ModelConfig
from mmocr_config_writers.dataset_config import DatasetConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare config files for a model')
    parser.add_argument('config', help='Preparation configuration file')
    parser.add_argument('-nd', '--no-dataset', action='store_true', default=False, help="Do not prepare dataset")
    parser.add_argument('-ndm','--no-det-model', action='store_true', default=False, help="Do not prepare detection model")
    parser.add_argument('-nrm','--no-recog-model', action='store_true', default=False, help="Do not prepare recognition model")
    args = parser.parse_args()

    return args

def processDatset(dataset_dict):

    print(f"{'=' * 30}Preparing Dataset{'=' * 30}")
    #Prepare dataset
    dataset_type = dataset_dict["type"]
    assert (dataset_type in ids.ids.keys()), f"{dataset_type} not available\n Available dataset descriptions include: {ids.ids.keys()}\n " \
                                             f"If a custom dataset has been created, include reference to that class " \
                                             f"in dataset_ids.py and __init__.py files in dataset_lib folder "

    dataset = ids.ids[dataset_type](**dataset_dict["init_params"])

    splits = list(dataset_dict["prepare_params"].keys())

    fnames = dict(textdet=[], textrecog=[])

    for split in splits:
        if not(dataset_dict["prepare_params"][split] is None):
            print(f"{'=' * 60}")
            print(f"Preparing dataset for {split}")
            prep_dict = dataset.process(**dataset_dict["prepare_params"][split])
            print(f"Preparing configuration files and cropping images...")
            out = dataset(prep_dict, split)
            fnames['textdet'].extend(out['det'])
            fnames['textrecog'].extend(out['recog'])
            print(f"Finished preparing for {split}")
            print(f"{'=' * 60}")
        else:
            print(f"Skipping {split} preparation")
    
    if dataset.save_dir.split("/")[0] == ".":
        config_save_dir = os.path.join(os.getcwd(), *dataset.save_dir.split("/")[1:])
    else:
        config_save_dir = dataset.save_dir
    
    for task in fnames.keys():
        config_writer = DatasetConfig(dataset.name, task, dataset_dict["config_save_dir"])
        
        config_writer(config_save_dir, fnames[task], splits)

def processModel(task_dict, task):

    if not(task_dict is None):
        print(f"Preparing model for {task}")
        model_config = ModelConfig(task, **task_dict)

        config = model_config()

        save_path = config()

        print(f"{task} configuration file can be found in : {save_path}", flush=True)

def main():
    args = parse_args()

    config = Config.fromfile(args.config)
    dataset_dict = config.dataset_dict
    det_model_dict = config.det_model_dict
    recog_model_dict = config.recog_model_dict

    #Prepare Dataset
    if not(dataset_dict is None) and not(args.no_dataset):
        if isinstance(dataset_dict, dict):
            dataset_dict = [dataset_dict]

        for dataset in dataset_dict:
            processDatset(dataset)

    #Prepare Model
    print(f"{'=' * 30}Preparing Model Configurations{'=' * 30}")

    if not(args.no_det_model):
        processModel(det_model_dict, "textdet")

    if not(args.no_recog_model):
        processModel(recog_model_dict, "textrecog")

    print(f"{'=' * 30}Finished preparing{'=' * 30}")

if __name__ == "__main__":
    main()






