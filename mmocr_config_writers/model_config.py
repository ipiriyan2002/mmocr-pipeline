from mmocr_config_writers.base_config import BaseConfig
from utils.code_block import *
from mmocr_config_writers.textdet_model import TextDetModelConfig
from mmocr_config_writers.textrec_model import TextRecModelConfig
import os

class ModelConfig(BaseConfig):

    DEFAULT_MODELS = dict(textdet=["dbnet", "dbnetpp", "drrg", "fcenet", "mask-rcnn", "panet", "psenet", "textsnake"],
                          textrecog=["abinet", "aster", "crnn", "master", "nrtr", "robust_scanner", "sar", "satrn", "svtr"])

    def __init__(self,
                 task,
                 train_datasets, val_datasets, test_datasets,
                 model,
                 backbone=None,
                 neck=None,
                 base=None,
                 epochs=1200,
                 schedule=None,
                 has_val=False,
                 train_batch_size=16,
                 test_batch_size=1, contents=dict()):

        super().__init__(task)

        self.task = task
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        assert (model in self.DEFAULT_MODELS[task])

        self.model = model
        self.backbone = backbone
        self.base = base
        self.neck = neck
        self.epochs = epochs
        self.schedule = schedule
        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.contents = contents
        


    def getSavePath(self, save_path):
        pass

    def __call__(self):

        if self.task == "textdet":

            return TextDetModelConfig(self.train_datasets,self.val_datasets, self.test_datasets, self.model, self.backbone,
                                      self.neck, self.epochs, self.schedule,
                                      self.has_val, self.train_batch_size, self.test_batch_size, self.contents)

        else:
            return TextRecModelConfig(self.train_datasets,self.val_datasets, self.test_datasets, self.model, self.base, self.epochs, self.schedule
                                      ,self.has_val, self.train_batch_size, self.test_batch_size, self.contents)
    