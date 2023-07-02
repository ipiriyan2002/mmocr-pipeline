import os
import json
from PIL import Image
import pandas as pd
from dataset_lib.mmocr_dataset import MMOCRDataset

class MemeDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None, generator=None):
        super().__init__(name, tasks, save_dir, generator)

    def readPath(self, ann_path, img_paths, split):
        datadict = {}

        with jsonlines.open(ann_path) as f:
            for line in f.iter():

                valid_path = []
                for img_path in img_paths:
                    pos_path = os.path.join(img_path, line["img"])
                    if os.path.exists(pos_path):
                        valid_path.append(pos_path)

                try:
                    valid_path = valid_path[0]
                except:
                    raise FileNotFoundError(f"No available valid path for image: {line['img']}")

                text = line["text"]
                classes = line['label'] if split in ["train", "dev"] else -1

                try:
                    pred_box = self.generator(valid_path, text)

                    instances = [
                        dict(text=v["original"], bbox=v["box"], ignore=v["ignore"], label=classes) for k, v in pred_box.items() if v["box"] != []
                    ]

                    if len(instances) >= 1:
                        datadict[line["img"]] = dict(img=valid_path, instances=instances)
                except:
                    continue

        return datadict


    def process(self, img_paths, ann_paths, split):
        assert isinstance(ann_paths, (str, list)), "Provide a list of annotation paths or a singular path"
        assert isinstance(img_paths, (str, list)), "Provide a list of image paths or a singular path"
        
        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        if isinstance(img_paths, str):
            img_paths = [img_paths]


        datadict = {}

        filtered_ann_paths = []

        for ann in ann_paths:
            if os.path.isdir(ann):
                for file in os.listdir(ann):

                    if ".jsonl" in file:
                        filtered_ann_paths.append(os.path.join(ann, file))
            else:
                filtered_ann_paths.append(ann)

        for ann_path in filtered_ann_paths:
            datadict.update(self.readPath(ann_path, img_paths, split))

        return datadict

