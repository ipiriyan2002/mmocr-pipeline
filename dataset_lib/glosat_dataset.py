import os
import json
from PIL import Image
import pandas as pd
from dataset_lib.mmocr_dataset import MMOCRDataset
from utils.box_translator_utils import *

class GlosatDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None, use_gen=False,generator=None):
        super().__init__(name, tasks, save_dir, use_gen, generator)


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

    """
    Load annotation paths
    """
    def loadAnns(self, ann_paths):

        ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]

        #Load and concat all csv files into one dataset
        dataset_list = [pd.read_csv(ann) for ann in ann_paths]
        dataset = pd.concat(dataset_list, ignore_index=True)

        #Get annotation and image names
        return dataset[["annotation","image_name"]]

    """
    Correct annotations are loaded and then overwrite previous dataset
    """
    def correctAnns(self, dataset, correct_anns):

        dataset_list = [pd.read_csv(ann) for ann in correct_anns]

        correct_dataset = pd.concat(dataset_list, ignore_index=True)
        correct_dataset = correct_dataset[["annotation","image_name"]]
        #Overwrite image_names
        #updated = pd.concat(
        #    [dataset, correct_dataset]
        #).drop_duplicates(['image_name'], keep='last', ignore_index=True).sort_values('image_name')

        merged = dataset.merge(correct_dataset,on="image_name", how="left", suffixes=("", "_correct"))
        merged["annotation"] = merged["annotation_correct"].fillna(merged["annotation"])
        updated = merged.drop(columns="annotation_correct")

        return updated

    """
    Get the image path given a image_name
    
    Assumes there is one valid pathway
    """
    def getImagePath(self, img_paths, image_name):

        img_paths = img_paths if isinstance(img_paths, list) else [img_paths]

        possible_paths = [os.path.join(img_path, image_name) for img_path in img_paths]

        filtered_paths = [path for path in possible_paths if os.path.exists(path)]

        if len(filtered_paths) != 1:
            raise ValueError(f"Expected 1 Valid image path but have received {len(filtered_paths)}")

        return filtered_paths[0]

    """
    Group multiple instance of a single images into a single dataset
    """
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
        assert not(img_paths is None), "Provide atleast one image path"
        assert (isinstance(img_paths, (str, list))), "Expected a string or a,list of strings for image paths"
        assert not(ann_paths is None), "Provide atleast one annotation path"
        assert (isinstance(ann_paths, (str, list))), "Expected a string or a,list of strings for annotation paths"

        ann_paths = ann_paths if isinstance(ann_paths, list) else [ann_paths]

        #Get definite annotations
        def_anns = []
        #Get corrected annotations
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


        #Load definite annotations
        data = self.loadAnns(def_anns)

        #If correct annotations exits, overwrite correct annotations
        if len(correct_anns) >= 1:
            data = self.correctAnns(data, correct_anns)

        #Group the multiple instances of images
        data = self.groupImageData(data)
        data_dict = {}

        #For all fnames
        for index, fname in enumerate(data["filename"]):

            #Get abstract data points
            abs_instances = [self.abstractDataDict(ann) for ann in data["grouped_anns"][index]]

            try:
                image_path = self.getImagePath(img_paths, f"{fname}.jpg")
            except:
                continue

            data_dict[fname] = dict(img=image_path, instances=abs_instances)

        return data_dict

    def process_multi(self, img_paths=None, ann_paths=None, split=None):
        assert isinstance(split, dict), "Expected a dictionary of split percentages for splits: train, test and val"

        data_dict = self.process(img_paths, ann_paths, split)

        split_percents = [split['train'], split['test'], split['val']]

        data_dicts = self.traintestsplit(data_dict, split_percents)

        return data_dicts


