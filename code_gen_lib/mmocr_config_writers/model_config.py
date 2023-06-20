from code_gen_lib.mmocr_config_writers.base_config import BaseConfig
from code_gen_lib.code_block import *
from code_gen_lib.mmocr_config_writers.textdet_model import TextDetModelConfig
from code_gen_lib.mmocr_config_writers.textrec_model import TextRecModelConfig
import os

class ModelConfig(BaseConfig):

    DEFAULT_MODELS = dict(textdet=["dbnet", "dbnetpp", "drrg", "fcenet", "mask-rcnn", "panet", "psenet", "textsnake"],
                          textrecog=["abinet", "aster", "crnn", "master", "nrtr", "robust_scanner", "sar", "satrn", "svtr"])

    def __init__(self,
                 task,
                 dataset,
                 model,
                 backbone=None,
                 neck=None,
                 base=None,
                 epochs=1200,
                 schedule=None,
                 has_val=False,
                 train_batch_size=16,
                 test_batch_size=1):

        super().__init__(dataset, task)

        self.task = task
        self.dataset = dataset

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
        


    def getSavePath(self, save_path):
        pass

    def __call__(self):

        if self.task == "textdet":

            return TextDetModelConfig(self.dataset, self.model, self.backbone,
                                      self.neck, self.epochs, self.schedule,
                                      self.has_val, self.train_batch_size, self.test_batch_size)

        else:
            return TextRecModelConfig(self.dataset, self.model, self.base, self.epochs, self.schedule
                                      ,self.has_val, self.train_batch_size, self.test_batch_size)
    