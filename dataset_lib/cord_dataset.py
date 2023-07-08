import io
import os
import json
import pandas as pd
from PIL import Image
from dataset_lib.mmocr_dataset import MMOCRDataset
from datasets import load_dataset

class CordDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None, use_gen=False, generator=None):
        super().__init__(name, tasks, save_dir, use_gen,generator)

        #For loading from huggingface library
        self.hugging_face_path = "naver-clova-ix/cord-v2"

    """
    Get the list of paths to append
    """
    def paths2Append(self, paths):

        if (paths is None):
            return []

        assert (isinstance(paths, (str, list))), f"Expected paths to be either a string or list of strings, but got {type(paths)}"

        return paths if isinstance(paths, list) else [paths]

    """
    Load the dataset, whether download from online or offline (given parquet files or folder containing them)
    """
    def loadDataset(self, img_paths, ann_paths, split):

        if (img_paths is None) and (ann_paths is None):
            #Load from hugging face library
            dataset = self.loadCordDataset(split)
        else:
            #Load from parquet files
            paths = []
            paths.extend(self.paths2Append(img_paths))
            paths.extend(self.paths2Append(ann_paths))

            filtered_paths = []

            for path in paths:
                #If in a folder
                if os.path.isdir(path):
                    for file in os.listdir(path):
                        if ".parquet" in file:
                            filtered_paths.append(os.path.join(path, file))

                elif os.path.exist(path) and (".parquet") in path:
                        filtered_paths.append(path)
                else:
                    pass

            #Load from parquet files
            if len(filtered_paths) > 0:
                dataset = self.loadParqFiles(filtered_paths)
            else:
                raise ValueError("Do not have sufficient information for load dataset")

        return dataset

    """
    Function for transforming bytes into PIL image object
    """
    def bytes2Image(self, x):
        return Image.open(io.BytesIO(x))

    """
    Load parquet files
    """
    def loadParqFiles(self, paths):
        dataset_list = []

        for parqfile in paths:
            dataset_list.append(pd.read_parquet(parqfile))

        #Join all parquet files together
        dataset = pd.concat(dataset_list, ignore_index=True)
        dataset["image"] = dataset["image"].apply(self.bytes2Image)

        return dataset
    """
    Loads the cord dataset using the hugging face library
    """
    def loadCordDataset(self, split):
        if split == "val":
            split = "validation"
        dataset = load_dataset(self.hugging_face_path, split=split)

        return dataset

    """
    Download images given the path to save, the images to download and ground truth
    """
    def downloadImages(self, path, images, gts, file_prefix="IMG_"):
        for index, image in enumerate(images):
            #Using ground truth to get id
            gt = eval(gts[index])
            gt_id = gt['meta']['image_id']

            #File name to save
            img_name = file_prefix + "{0}.jpg".format(gt_id)
            #Save image
            image.save(os.path.join(path,img_name))


    """
    Unpack ground truth into the datadict containing image name and instances
    """
    def getDataDict(self, path, gts, file_prefix="IMG_"):

        data_dict = {}

        for gt in gts:
            gt_eval = eval(gt)
            #File Path
            img_name = file_prefix + "{0}.jpg".format(gt_eval['meta']['image_id'])
            final_img_name = '/'.join(os.path.join(path, img_name).split('/')[-4:])

            #Instances
            instances = []
            for vl in gt_eval['valid_line']:
                words = vl['words']
                #Load all words
                for word in words:
                    quad = word['quad']
                    bbox = [quad['x1'],quad['y1'],quad['x3'],quad['y3']]
                    inst_dict = dict(bbox=bbox, text=word["text"], ignore=False)
                    instances.append(inst_dict)


            data_dict[img_name] = dict(img=final_img_name, instances=instances)

        return data_dict


    def process(self, img_paths=None, ann_paths =None, split=None):
        """
        Prepares a data_dict for further json creation
        :param csv_path:
        :param img_path:
        :return:
        """
        assert not(split is None), f"Provide a dataset split"
        assert (isinstance(split, str)), f"Expected type of split to be either string, but received {type(split)}"

        dataset = self.loadDataset(img_paths, ann_paths, split)
        path = os.path.join(self.save_dir, f"images/Det/{split}")

        if not(os.path.exists(path)):
            os.makedirs(path)

        self.downloadImages(path, dataset['image'], dataset['ground_truth'])
        data_dict = self.getDataDict(path, dataset['ground_truth'])

        return data_dict

    def process_multi(self, img_paths=None, ann_paths=None, split=None):

        assert (isinstance(img_paths, list) or isinstance(ann_paths, list)), "Expected parquet file pathways as input for either img_paths or ann_paths"
        assert isinstance(split, dict), "Expected a dictionary of split percentages for splits: train, test and val"

        #Need paths to parquet files in either img_paths or ann_paths
        img_paths = img_paths if not(img_paths is None) else []
        ann_paths = ann_paths if not(ann_paths is None) else []

        #Load dataset
        dataset = self.loadDataset(img_paths, ann_paths, split)
        path = os.path.join(self.save_dir, f"images/Det/")

        if not(os.path.exists(path)):
            os.makedirs(path)

        #Download images and get datadict
        self.downloadImages(path, dataset['image'], dataset['ground_truth'])
        data_dict = self.getDataDict(path, dataset['ground_truth'])

        #Get split percents
        split_percents = [split['train'], split['test'], split['val']]

        #Seperate into train, test and validation split
        data_dicts = self.traintestsplit(data_dict, split_percents)

        return data_dicts




