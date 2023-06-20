import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

#Custom Imports
from dataset_lib import dataset_ids as ids
from code_gen_lib.mmocr_config_writers.model_config import ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare config files for a model')
    parser.add_argument('config', help='Preparation configuration file')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    dataset_dict = args.config.dataset_dict
    model_dict = args.config.model_dict

    assert not(dataset_dict is None), "Dataset configuration not providied"
    assert not(model_dict is None), "Model configuration not provided"

    #Prepare dataset
    dataset_type = dataset_dict["type"]
    assert (dataset_type in ids.ids.keys()), f"{dataset_type} not available\n Available dataset descriptions include: {ids.ids.keys()}\n " \
                                             f"If a custom dataset has been created, include reference to that class " \
                                             f"in dataset_ids.py and __init__.py files in dataset_lib folder "
    dataset = ids.ids[dataset_type](*dataset_dict["init_params"])

    splits = list(dataset_dict["prepare_params"].keys())

    for split in splits:
        if not(split is None):
            prep_dict = dataset.prepare(*dataset_dict["prepare_params"][split])

            dataset(prep_dict, split)

    #Prepare Model

    tasks = ["textdet", "textrecog"]

    for task in tasks:
        task_dict = model_dict[task]

        if not(task_dict is None):
            model_config = ModelConfig(task, *task_dict)

            config = model_config()

            save_path = config()

            print(f"{task} configuration file can be found in : {save_path}", flush=True)



if __name__ == "__main__":
    main()






