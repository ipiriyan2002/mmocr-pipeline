from utils.box_translator_utils import voc2quad
import os
import json
from PIL import Image

class MMOCRDataset:

    DEF_TASK_NAMES = ["det", "recog"]

    def __init__(self, name, tasks, save_dir=None):

        assert isinstance(tasks, list), f"Expected tasks to be a list, got {type(tasks)}"

        #Name of Dataset
        self.name = name

        #Tasks
        self.isDet = False
        self.isRecog = False
        self.checkTasks(tasks)

        #Save Directory
        if save_dir is None:
            if os.path.exists("../mmocr/data/"):
                self.save_dir = os.path.join("../mmocr/data/", f"{name}/")
            else:
                self.save_dir = os.path.join("../data/", f"{name}/")
        else:
            self.save_dir = save_dir

        try:
            os.makedirs(self.save_dir)
        except:
            pass
            

    """
    Check if given tasks is possible
    """
    def checkTasks(self, tasks):

        for task in tasks:

            assert (task.lower() in self.DEF_TASK_NAMES), f"Expected tasks {self.DEF_TASK_NAMES}, but received {task.lower()}"

            if task.lower() == "det": self.isDet = True
            elif task.lower() == "recog": self.isRecog = True


    def saveJson(self, final_dict, split, task):

        save_path = os.path.join(self.save_dir, f"text{task}_{split}.json")

        with open(save_path, "w") as f:
            json.dump(final_dict, f)

    def getAbstractInstance(self, img_dict):

        #All instances
        instances = []

        #Get all instance dicts
        for inst in img_dict["instances"]:
            #Bounding box, polygon, text
            box = inst["bbox"]
            text = inst["text"]
            poly = voc2quad(box)

            #Ignore Key
            if "ignore" in inst.keys():
                ignore = inst["ignore"]
            else:
                ignore = 0

            instances.append(dict(text=text, bbox=box, bbox_label=0, polygon=poly, ignore=ignore))

        return instances

    def createDetJsonFile(self, data_dict, split):

        #meta info
        meta_info_default = dict(dataset_type="TextDetDataset", task_name="text_det", category=[dict(id=0, name="text")])

        #all datalists
        data_list = []

        for index, img in enumerate(data_dict.keys()):

            img_dict = data_dict[img]

            open_img = Image.open(os.path.join(self.save_dir,img_dict["img"]))
            width, height = open_img.size

            instances = self.getAbstractInstance(img_dict)

            data_list.append(dict(img_path=img_dict["img"], height=height, width=width, instances=instances))

        #final dict for detection task
        final_dict = dict(metainfo=meta_info_default, data_list=data_list)
        self.saveJson(final_dict, split, "det")

    def retreiveCropsDict(self, img_name, img_path, anns, save_dir):


        open_img = Image.open(os.path.join(self.save_dir,img_path))

        all_crops = []

        for index, ann_dict in enumerate(anns):
            img_name = img_name.split(".")[0]
            save_img_name = os.path.join(save_dir,f"{img_name}_{index}.jpg")
            
            try:
                open_img.crop(ann_dict["bbox"]).save(save_img_name)
            except:
                continue

            img_path = '/'.join(save_img_name.split("/")[-4:])
            out_dict = dict(img_path=img_path, instances=list(dict(text=ann_dict["text"])))

            all_crops.append(out_dict)

        return all_crops

    def createRecogJsonFile(self, data_dict, split):

        #meta info
        meta_info_default = dict(dataset_type="TextRecogDataset", task_name="textrecog")

        #all datalists
        data_list = []

        #Save path for crops
        save_crop_path = os.path.join(self.save_dir, f"images/Recog/{split}/")

        try:
            os.makedirs(save_crop_path)
        except:
            pass

        print("Croping and Downloading for recognition task")
        for index, img in enumerate(data_dict.keys()):

            img_dict = data_dict[img]

            instances = self.getAbstractInstance(img_dict)

            all_img_crops = self.retreiveCropsDict(img, img_dict["img"], instances, save_crop_path)

            data_list.extend(all_img_crops)
        print("Croping and Downloading finished")
        #final dict for detection task
        final_dict = dict(metainfo=meta_info_default, data_list=data_list)
        self.saveJson(final_dict, split, "recog")

    def __call__(self, data_dict, split):
        """
        Assumes data_dict is of format:
        {<<img_name>> : {
            img:<<img_path>>,
            instances: [{bbox:[x1,y1,x3,y3], text:<<text at bbox loc>>, <<optional>>ignore:Boolean}...]
            }
        }
        :param data_dict:
        :param split: split name
        :return:
        """

        if self.isDet:
            print("Preparing JSON file for detection task")
            self.createDetJsonFile(data_dict, split)

        if self.isRecog:
            print("Preparing JSON file for recognition task")
            self.createRecogJsonFile(data_dict, split)