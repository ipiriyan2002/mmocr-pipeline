import os
import json
from PIL import Image
from dataset_lib.mmocr_dataset import MMOCRDataset
from datasets import load_dataset

class CordDataset(MMOCRDataset):

    def __init__(self, name, tasks, save_dir=None):
        super().__init__(name, tasks, save_dir)

        self.hugging_face_path = "naver-clova-ix/cord-v2"

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

    def prepare(self, split):
        """
        Prepares a data_dict for further json creation
        :param csv_path:
        :param img_path:
        :return:
        """
        assert (isinstance(split, str)), f"Expected type of split to be either string, but received {type(split)}"

        dataset = self.loadCordDataset(split)
        path = os.path.join(self.save_dir, f"images/Det/{split}")

        if not(os.path.exists(path)):
            os.makedirs(path)

        self.downloadImages(path, dataset['image'], dataset['ground_truth'])
        data_dict = self.getDataDict(path, dataset['ground_truth'])

        return data_dict

