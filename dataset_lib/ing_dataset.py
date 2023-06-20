import os
import json
from PIL import Image
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

    def prepare(self, csv_path, img_path):
        """
        Prepares a data_dict for further json creation
        :param csv_path:
        :param img_path:
        :return:
        """

        #Get the data
        data = pd.read_csv(csv_path)
        data = data[["annotation","filename"]]

        data_dict = {}

        for index, fname in data["filename"]:

            instances = [self.abstractDataDict(ann) for ann in eval(data["annotation"][index])]

            image_path = os.path.join(img_path, f"{fname}.jpg")

            
            data_dict[fname] = dict(img=image_path, instances=instances)

        return data_dict

