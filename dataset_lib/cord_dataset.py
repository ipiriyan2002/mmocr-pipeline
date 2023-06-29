import io
import os
import json
import pandas as pd
from PIL import Image
from dataset_lib.mmocr_dataset import MMOCRDataset
from datasets import load_dataset

class CordDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None, generator=None):
        super().__init__(name, tasks, save_dir, generator)

        self.hugging_face_path = "naver-clova-ix/cord-v2"


    def paths2Append(self, paths):

        if (paths is None):
            return []

        assert (isinstance(paths, (str, list))), f"Expected paths to be either a string or list of strings, but got {type(paths)}"

        return paths if isinstance(paths, list) else [paths]

    def loadDataset(self, img_paths, ann_paths, split):

        if (img_paths is None) and (ann_paths is None):

            dataset = self.loadCordDataset(split)
        else:
            paths = []
            paths.extend(self.paths2Append(img_paths))
            paths.extend(self.paths2Append(ann_paths))

            filtered_paths = [path for path in paths if path.split(".")[-1] == "parquet"]

            if len(filtered_paths) > 0:
                dataset = self.loadParqFiles(filtered_paths)
            else:
                raise ValueError("Do not have sufficient information for load dataset")

    def bytes2Image(self, x):
        return Image.open(io.BytesIO(x))

    def loadParqFiles(self, paths):
        dataset_list = []

        for parqfile in paths:
            dataset_list.append(pd.read_parquet(parqfile))

        dataset = pd.concat(dataset_list, ignore_index=True)
        dataset["image"] = dataset["image"].apply(self.bytes2Image)

        return dataset

    #Loads the cord dataset using the hugging face library
    def loadCordDataset(self, split):
        if split == "val":
            split = "validation"
        dataset = load_dataset(self.hugging_face_path, split=split)

        return dataset

    def downloadImages(self, path, images, gts, file_prefix="IMG_"):
        for index, image in enumerate(images):
            gt = eval(gts[index])
            gt_id = gt['meta']['image_id']
            img_name = file_prefix + "{0}.jpg".format(gt_id)
            image.save(os.path.join(path,img_name))

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
                for word in words:
                    quad = word['quad']
                    bbox = [quad['x1'],quad['y1'],quad['x3'],quad['y3']]
                    inst_dict = dict(bbox=bbox, text=word["text"], ignore=False)
                    instances.append(inst_dict)


            data_dict[img_name] = dict(img=final_img_name, instances=instances)

        return data_dict

    def prepare(self, img_paths=None, ann_paths =None, split=None):
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

