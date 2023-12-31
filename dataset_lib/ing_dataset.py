import os
import json
from PIL import Image
import pandas as pd
from dataset_lib.mmocr_dataset import MMOCRDataset
from utils.box_translator_utils import *


class IngDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None, use_gen=True,generator=None):
        super().__init__(name, tasks, save_dir, use_gen, generator)

    """
    Perform basic text abstraction such as removing line delimiters and non-training characters (for now: _@_)
    """

    def abstractText(self, text):
        # text = text.replace("\n", " ")
        text = text.replace("_@_", "")

        return text.strip()

    """
    Read a single instance of an image given the format of the Ingredient lists,
    Return an abstract version for MMOCR
    """

    def abstractDataDict(self, ann_dict):
        # Abstract format of text (By doing some basic preprocessing)
        text = self.abstractText(ann_dict["metadata"]['shapeTranscription']['text'])
        # Get bounding box and polygon
        box = coco2voc(ann_dict['coordinates'])

        # By default, ignore any angles that is not 0
        ignore = ann_dict['angle'] != 0

        # Update the key value pairs
        output = dict(text=text, bbox=box, ignore=ignore)

        return output

    """
    Load annotation paths
    """
    def loadAnns(self, ann_paths):

        ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]

        filtered_paths = []

        #Load files from folders
        for ann in ann_paths:
            if os.path.isdir(ann):
                for file in os.listdir(ann):

                    if ".csv" in file:
                        filtered_paths.append(os.path.join(ann, file))
            else:
                filtered_paths.append(ann)

        #Join together the different csv files and then return annotation and filenames
        dataset_list = [pd.read_csv(ann) for ann in filtered_paths]
        dataset = pd.concat(dataset_list, ignore_index=True)

        return dataset[["annotation", "filename"]]

    """
    Get the valid image path for any given image name
    """
    def getImagePath(self, img_paths, image_name):

        img_paths = img_paths if isinstance(img_paths, list) else [img_paths]

        possible_paths = [os.path.join(img_path, image_name) for img_path in img_paths]

        filtered_paths = [path for path in possible_paths if os.path.exists(path)]

        if len(filtered_paths) != 1:
            raise ValueError(f"Expected 1 Valid image path but have received {len(filtered_paths)}")

        return filtered_paths[0]

    def process(self, img_paths, ann_paths, split):
        """
        Prepares a data_dict for further json creation
        """

        # Get the data
        assert not (img_paths is None), "Provide atleast one image path"
        assert (isinstance(img_paths, (str, list))), "Expected a string or a,list of strings for image paths"
        assert not (ann_paths is None), "Provide atleast one annotation path"
        assert (isinstance(ann_paths, (str, list))), "Expected a string or a,list of strings for annotation paths"

        #Load annotations
        data = self.loadAnns(ann_paths)

        #get the data in a dict form
        data_dict = {}

        for index, fname in enumerate(data["filename"]):

            #Abstract instances
            abs_instances = [self.abstractDataDict(ann) for ann in eval(data["annotation"][index])]

            try:
                image_path = self.getImagePath(img_paths, f"{fname}.jpg")
            except:
                continue

            #Get dense texts and boxes
            crop_texts = []
            crop_boxes = []

            not_added = []
            for abs_inst in abs_instances:

                if ("\n" in abs_inst["text"]):
                    crop_boxes.append(abs_inst["bbox"])
                    crop_texts.append(abs_inst["text"])
                else:
                    not_added.append(abs_inst)

            #use MMOCR generator to generate words for localised crops
            instances = []
            if len(crop_texts) > 0:
                crop_boxes = crop_boxes if not (crop_boxes == []) else None
                out = self.generator(image_path, crop_texts, crop_boxes)

                instances.extend([dict(text=v['original'], bbox=v['box'], ignore=v['ignore']) for k, v in out.items() if
                                  not (v['box'] == [])])

            instances.extend(not_added)

            data_dict[fname] = dict(img=image_path, instances=instances)

        return data_dict

    def process_multi(self, img_paths=None, ann_paths=None, split=None):
        assert isinstance(split, dict), "Expected a dictionary of split percentages for splits: train, test and val"

        data_dict = self.process(img_paths, ann_paths, split)

        split_percents = [split['train'], split['test'], split['val']]

        data_dicts = self.traintestsplit(data_dict, split_percents)

        return data_dicts
