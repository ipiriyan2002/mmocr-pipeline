from utils.box_translator_utils import voc2quad
import os
import json
from PIL import Image
from abc import ABC, abstractmethod
from bbox_gen.generator import MMOCRBoxGenerator
import random, math

"""
Base class for dataset processing, including JSON file creation for MMOCR training
"""
class MMOCRDataset(ABC):
    DEF_TASK_NAMES = ["det", "recog"]


    def __init__(self, name, tasks, save_dir=None, use_gen=True, generator=None):

        #Makes sure defined tasks is a list of tasks
        assert isinstance(tasks, list), f"Expected tasks to be a list, got {type(tasks)}"

        # Name of Dataset
        self.name = name

        # Tasks
        self.isDet = False
        self.isRecog = False
        self.checkTasks(tasks)

        # Save Directory
        if save_dir is None:
            if os.path.exists("../mmocr/data/"):
                self.save_dir = os.path.join("../mmocr/data/", f"{name}/")
            elif os.path.exists("../data/"):
                self.save_dir = os.path.join("../data/", f"{name}/")
            else:
                self.save_dir = os.path.join("./data/", f"{name}/")
        else:
            self.save_dir = save_dir

        try:
            os.makedirs(self.save_dir)
        except:
            pass


        #get the default generator if generator is None
        #else customise the generator according to params defined
        if use_gen:
            if generator is None:
                self.generator = MMOCRBoxGenerator()
            else:
                self.generator = MMOCRBoxGenerator(**generator)

    """
    Check if given tasks is possible
    """
    def checkTasks(self, tasks):

        for task in tasks:

            assert (
                    task.lower() in self.DEF_TASK_NAMES), f"Expected tasks {self.DEF_TASK_NAMES}, but received {task.lower()}"


            if task.lower() == "det":
                self.isDet = True
            elif task.lower() == "recog":
                self.isRecog = True

    """
    Save the data in a JSON format file and return the file name of saved json file
    """
    def saveJson(self, final_dict, split, task):

        fname = f"text{task}_{split}.json"
        save_path = os.path.join(self.save_dir, fname)

        with open(save_path, "w") as f:
            json.dump(final_dict, f)

        print(f"JSON file for MMOCR task {task} can be found in: {save_path}")
        return fname

    """
    Get an abstract list of instances given a data dict for an image
    """
    def getAbstractInstance(self, img_dict):

        # All instances
        instances = []

        # Get all instance dicts
        for inst in img_dict["instances"]:
            # Bounding box, polygon, text
            box = inst["bbox"]
            text = inst["text"]
            poly = voc2quad(box)

            # Ignore Key
            if "ignore" in inst.keys():
                ignore = inst["ignore"]
            else:
                ignore = False

            #Relays the original text, bounding box (voc format and polygon) of that text, the label and whether to ignore this instance
            instances.append(dict(text=text, bbox=box, bbox_label=0, polygon=poly, ignore=ignore))

        return instances

    """
    Create JSON file for detection
    """
    def createDetJsonFile(self, data_dict, split):

        # meta info
        meta_info_default = dict(dataset_type="TextDetDataset", task_name="text_det",
                                 category=[dict(id=0, name="text")])

        # all datalists
        data_list = []


        for index, img in enumerate(data_dict.keys()):
            img_dict = data_dict[img]

            """
            Try opening the image for following cases:
            
            Case 1: absolute path not given
            Case 2: absolute path given
            """
            try:
                open_img = Image.open(os.path.join(self.save_dir, img_dict["img"]))
            except:
                open_img = Image.open(img_dict["img"])

            #Get image size
            width, height = open_img.size

            #All instances for said image in an abstracted form (the form for detection)
            instances = self.getAbstractInstance(img_dict)

            #add image data to data_list
            data_list.append(dict(img_path=img_dict["img"], height=height, width=width, instances=instances))

        # final dict for detection task
        final_dict = dict(metainfo=meta_info_default, data_list=data_list)
        fname = self.saveJson(final_dict, split, "det")

        return fname

    """
    Given an image, crop all instances appearing in that image and save all crops at save dir
    
    Also returns a dictionary of recognition training data for image 
    """
    def retreiveCropsDict(self, img_name, img_path, anns, save_dir):

        if os.path.exists(img_path):
            img_path = img_path
        else:
            img_path = os.path.join(self.save_dir, img_path)
        open_img = Image.open(img_path)

        all_crops = []

        for index, ann_dict in enumerate(anns):
            img_name = img_name.split(".")[0]
            save_img_name = os.path.join(save_dir, f"{img_name}_{index}.jpg")
            if ann_dict['ignore']:
                save_img_name = os.path.join(save_dir, f"{img_name}_{index}_ERROR.jpg")

            try:
                open_img.crop(ann_dict["bbox"]).save(save_img_name)
            except:
                continue

            img_path = '/'.join(save_img_name.split("/")[-4:])
            
            if not(ann_dict['ignore']):
                out_dict = dict(
                    img_path=img_path,
                    instances=[dict(text=ann_dict["text"])]
                )

                all_crops.append(out_dict)

        return all_crops

    """
    Perform cropping and generate JSON file for recognition task
    """
    def createRecogJsonFile(self, data_dict, split):

        # meta info
        meta_info_default = dict(dataset_type="TextRecogDataset", task_name="textrecog")

        # all datalists
        data_list = []

        # Save path for crops
        save_crop_path = os.path.join(self.save_dir, f"images/Recog/{split}/")

        try:
            os.makedirs(save_crop_path)
        except:
            pass

        print("Croping and Downloading for recognition task")
        for index, img in enumerate(data_dict.keys()):
            img_dict = data_dict[img]

            #All instances of image
            instances = self.getAbstractInstance(img_dict)

            #Crops dict
            all_img_crops = self.retreiveCropsDict(img, img_dict["img"], instances, save_crop_path)

            data_list.extend(all_img_crops)
        print("Croping and Downloading finished")
        # final dict for detection task
        final_dict = dict(metainfo=meta_info_default, data_list=data_list)
        fname = self.saveJson(final_dict, split, "recog")

        return fname

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
        fnames = dict(det=[], recog=[])

        if self.isDet:
            print("Preparing JSON file for detection task")
            out = self.createDetJsonFile(data_dict, split)
            fnames["det"].append(out)

        if self.isRecog:
            print("Preparing JSON file for recognition task")
            out = self.createRecogJsonFile(data_dict, split)
            fnames['recog'].append(out)

        return fnames

    """
    Used to split a data dictionary into splits depending on a percentage
    
    where split_percent denotes the percentage for [train, test, val] in a list
    
    E.X:
     [0.8, 0.1, 0.1] -> 80% train, 10% test, 10% val
    """
    def traintestsplit(self, datadict, split_percent):
        keys = list(datadict.keys())
        random.shuffle(keys)


        train_split = math.ceil(split_percent[0] * len(keys))
        train_keys = keys[:train_split]

        
        test_split = train_split + math.ceil(split_percent[1] * len(keys))
        test_keys = keys[train_split:test_split]

        val_keys = keys[test_split:]
        
        train_dict = {k : datadict[k] for k in train_keys}
        test_dict = {k : datadict[k] for k in test_keys}
        val_dict = {k : datadict[k] for k in val_keys}

        return dict(train=train_dict, test=test_dict, val=val_dict)

    @abstractmethod
    def process(self, img_paths, ann_paths, split):
        pass

    @abstractmethod
    def process_multi(self, img_paths, ann_paths, split):
        pass