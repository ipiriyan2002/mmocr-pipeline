import os
import json
import jsonlines
from PIL import Image
import pandas as pd
from bbox_gen.generator import MMOCRBoxGenerator
from bbox_gen.utils.box_processing import order_sent
from dataset_lib.mmocr_dataset import MMOCRDataset


class MemeDataset(MMOCRDataset):

    def __init__(self, name, tasks,save_dir=None, use_gen=True, generator=None):
        super().__init__(name, tasks, save_dir, use_gen,generator)



    def readPath(self, ann_path, img_paths):

        if isinstance(img_paths, str):
            img_paths = [img_paths]

        datadict = {}

        with jsonlines.open(ann_path) as f:
            for line in f.iter(): #Read line by line
                img_name = line["img"].split("/")[-1]
                valid_path = []
                for img_path in img_paths:
                    pos_path = os.path.join(img_path, img_name)
                    if os.path.exists(pos_path):
                        valid_path.append(pos_path)

                #Get valid path
                try:
                    valid_path = valid_path[0]
                except:
                    raise FileNotFoundError(f"No available valid path for image: {img_name}")

                text = line["text"]

                try:
                    classes = line['label']
                except:
                    classes = -1

                try:
                    #Generate boxes for image
                    instances = self.process_single(valid_path, text, classes)

                    openedImg = Image.open(valid_path)
                    width, height = openedImg.size

                    if len(instances) >= 1:
                        dict2save = dict(img=valid_path, original=text, width=width, height=height, instances=instances)
                        datadict[line["img"]] = dict2save
                        #img_save_name = img_name.split(".")[0]
                        #fpath = os.path.join(self.save_dir, f"{img_save_name}.json")

                        #with open(fpath, "w") as f:
                        #    json.dump(dict2save, f)
                except:
                    continue

        return datadict

    """
    Process a single image, given text, generate bounding boxes
    """
    def process_single(self, image, text, classes=None):
        classes = classes if not(classes is None) else -1

        pred_box = self.generator(image, text)

        pred_box = order_sent(pred_box)

        instances = [
            dict(text=v["original"], bbox=v["box"], ignore=v["ignore"], label=classes) for _, v in pred_box.items() if v["box"] != []
        ]

        return instances


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

        #For all annotation paths, read path
        for ann_path in filtered_ann_paths:
            datadict.update(self.readPath(ann_path, img_paths))

        return datadict

    def process_multi(self, img_paths=None, ann_paths=None, split=None):
        assert isinstance(split, dict), "Expected a dictionary of split percentages for splits: train, test and val"

        data_dict = self.process(img_paths, ann_paths, split)

        split_percents = [split['train'], split['test'], split['val']]

        data_dicts = self.traintestsplit(data_dict, split_percents)

        return data_dicts

