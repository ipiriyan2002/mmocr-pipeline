import os
import json
from PIL import Image
import pandas as pd
from dataset_lib.mmocr_dataset import MMOCRDataset
from utils.box_translator_utils import *

class GlosatDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None, generator=None):
        super().__init__(name, tasks, save_dir, generator)


    """
    Perform basic text abstraction such as removing line delimiters and non-training characters (for now: _@_)
    """
    def abstractText(self, text):
        text = text.replace("_@_", "")

        return text.strip()

    """
    Read a single instance of an image given the format of the Ingredient lists,
    Return an abstract version for MMOCR
    """
    def abstractDataDict(self, ann_dict):
        #Abstract format of text (By doing some basic preprocessing)
        try:
            text = self.abstractText(ann_dict["metadata"]['shapeTranscription']['text'])
        except:
            text = ""
        #Get bounding box and polygon
        coords = ann_dict['coordinates']
        x, y , w, h = coords['x'], coords['y'], coords['w'], coords['h']
        box = coco2voc([x,y,w,h])

        #By default, ignore any angles that is not 0
        ignore = (text == "") or (text == " ") or (text == "-") or (text in ["@@@", "$$$", "###"])

        #Update the key value pairs
        output = dict(text=text, bbox=box, ignore=ignore)

        return output

    def loadAnns(self, ann_paths):

        ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]

        dataset_list = [pd.read_csv(ann) for ann in ann_paths]
        dataset = pd.concat(dataset_list, ignore_index=True)

        return dataset[["annotation","image_name"]]

    def correctAnns(self, dataset, correct_anns):

        dataset_list = [pd.read_csv(ann) for ann in correct_anns]

        correct_dataset = pd.concat(dataset_list, ignore_index=True)

        updated = pd.concat([dataset, correct_dataset]).drop_duplicates(['image_name']).sort_values('image_name')

        return updated

    def getImagePath(self, img_paths, image_name):

        img_paths = img_paths if isinstance(img_paths, list) else [img_paths]

        possible_paths = [os.path.join(img_path, image_name) for img_path in img_paths]

        filtered_paths = [path for path in possible_paths if os.path.exists(path)]

        if len(filtered_paths) != 1:
            raise ValueError(f"Expected 1 Valid image path but have received {len(filtered_paths)}")

        return filtered_paths[0]

    def groupImageData(self, dataset):
        joined_dict = {}

        for index, fname in enumerate(dataset['image_name']):

            fnum = fname.split("_")[0]

            if fnum in joined_dict.keys():
                joined_dict[fnum].extend(eval(dataset['annotation'][index]))
            else:
                joined_dict[fnum] = []
                joined_dict[fnum].extend(eval(dataset['annotation'][index]))

        return pd.DataFrame(data=dict(filename=list(joined_dict.keys()), grouped_anns=list(joined_dict.values())))

    def process(self, img_paths, ann_paths, split):
        """
        Prepares a data_dict for further json creation
        """

        #Get the data

        global abs_inst
        assert not(img_paths is None), "Provide atleast one image path"
        assert (isinstance(img_paths, (str, list))), "Expected a string or a,list of strings for image paths"
        assert not(ann_paths is None), "Provide atleast one annotation path"
        assert (isinstance(ann_paths, (str, list))), "Expected a string or a,list of strings for annotation paths"

        ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]
        def_anns = []
        correct_anns = []

        for ann in ann_paths:
            if os.path.isdir(ann):
                for file in os.listdir(ann):

                    if ".csv" in file:

                        if "correct" in file:
                            correct_anns.append(os.path.join(ann, file))
                        else:
                            def_anns.append(os.path.join(ann, file))
            else:
                if "correct" in ann:
                    correct_anns.append(ann)
                else:
                    def_anns.append(ann)


        data = self.loadAnns(def_anns)

        if len(correct_anns) > 1:
            data = self.correctAnns(data, correct_anns)

        data = self.groupImageData(data)
        data_dict = {}

        for index, fname in enumerate(data["filename"]):

            abs_instances = [self.abstractDataDict(ann) for ann in data["grouped_anns"][index]]

            try:
                image_path = self.getImagePath(img_paths, f"{fname}.jpg")
            except:
                continue

            data_dict[fname] = dict(img=image_path, instances=abs_instances)

        return data_dict

