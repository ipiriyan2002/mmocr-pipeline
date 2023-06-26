import os
import json
from PIL import Image
import pandas as pd
from dataset_lib.mmocr_dataset import MMOCRDataset

class IngDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None):
        super().__init__(name, tasks, save_dir)

    """
    Given a dictionary format of a COCO format, return VOC format bounding box
    """
    def getVocBox(self, box_dict):
        #Unpack data
        x_min = box_dict['x']
        y_min = box_dict['y']
        w = box_dict['w']
        h = box_dict['h']

        #Calculate VOC format box
        x_max = x_min + w
        y_max = y_min + h

        voc_box = [x_min, y_min, x_max, y_max]
        return voc_box

    """
    Perform basic text abstraction such as removing line delimiters and non-training characters (for now: _@_)
    """
    def abstractText(self, text):
        text = text.replace("\n", " ")
        text = text.replace("_@_", "")

        return text.strip()

    """
    Read a single instance of an image given the format of the Ingredient lists,
    Return an abstract version for MMOCR
    """
    def abstractDataDict(self, ann_dict):
        #Abstract format of text (By doing some basic preprocessing)
        text = self.abstractText(ann_dict["metadata"]['shapeTranscription']['text'])
        #Get bounding box and polygon
        box = self.getVocBox(ann_dict['coordinates'])

        #By default, ignore any angles that is not 0
        ignore = ann_dict['angle'] != 0

        #Update the key value pairs
        output = dict(text=text, bbox=box, ignore=ignore)

        return output

    def loadAnns(self, ann_paths):

        ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]

        dataset_list = [pd.read_csv(ann) for ann in ann_paths]
        dataset = pd.concat(dataset_list, ignore_index=True)

        return dataset[["annotation","filename"]]

    def getImagePath(self, img_paths, image_name):

        img_paths = img_paths if isinstance(img_paths, list) else [img_paths]

        possible_paths = [os.path.join(img_path, image_name) for img_path in img_paths]

        filtered_paths = [path for path in possible_paths if os.path.exists(path)]

        if len(filtered_paths)  != 1:
            raise ValueError(f"Expected 1 Valid image path but have received {len(filtered_paths)}")

        return filtered_paths[0]

    def prepare(self, img_paths, ann_paths, split):
        """
        Prepares a data_dict for further json creation
        """

        #Get the data

        assert not(img_paths is None), "Provide atleast one image path"
        assert (isinstance(img_paths, (str, list))), "Expected a string or a,list of strings for image paths"
        assert not(ann_paths is None), "Provide atleast one annotation path"
        assert (isinstance(ann_paths, (str, list))), "Expected a string or a,list of strings for annotation paths"

        data = self.loadAnns(ann_paths)

        data_dict = {}

        for index, fname in data["filename"]:

            instances = [self.abstractDataDict(ann) for ann in eval(data["annotation"][index])]

            image_path = self.getImagePath(img_paths, f"{fname}.jpg")
            
            data_dict[fname] = dict(img=image_path, instances=instances)

        return data_dict

