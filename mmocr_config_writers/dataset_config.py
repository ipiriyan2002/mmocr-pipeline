import os
from mmocr_config_writers.base_config import BaseConfig
from utils.code_block import *

"""
Create config files for datasets
"""
class DatasetConfig(BaseConfig):

    def __init__(self, dataset, task, save_path=None):
        self.dataset = dataset
        self.DEFAULT_PATH = os.path.join("configs", f"{task}", "_base_", "datasets")
        self.head = f"{self.dataset}_{task}"
        super().__init__(task, save_path)


    def generateSplitBlock(self,fname,split):

        assert (split in ["train", "val", "test"]), f"Expected one of the following ['train', 'val', 'test'], but got {split}"

        fname = fname if ".json" in fname else fname+".json"

        split_statement = "filter_cfg=dict(filter_empty_gt=True, min_size=32)" if split in ["train", "val"] else "test_mode=True"
        return StatementBlock(statements=[f"{self.head}_{split} = dict(", "\ttype='OCRDataset',", f"\tdata_root={self.head}_data_root,",
                                          f"\tann_file='{fname}',", f"\t{split_statement},", "\tpipeline=None)"])

    def getSavePath(self, save_path):

        if save_path is None:
            if os.path.exists(self.DEFAULT_PATH):
                return os.path.join(self.DEFAULT_PATH, f"{self.dataset}.py")
            elif os.path.exists(os.path.join("../mmocr", self.DEFAULT_PATH)):
                return os.path.join("../mmocr", self.DEFAULT_PATH, f"{self.dataset}.py")
            else:
                return f"./{self.dataset}.py"
        else:
            return save_path
        
    def __call__(self,json_dir, fnames, splits):

        statements = [f"{self.head}_data_root = '{json_dir}'", ""]

        for fname, split in zip(fnames, splits):

            statements.append(self.generateSplitBlock(fname, split))
            statements.append("")

        final_statement_block = StatementBlock(statements=statements)

        if ".py" in self.save_path:
            with open(self.save_path, "w", encoding='utf-8') as f:
                f.write(str(self))
        else:
            with open(self.save_path, "w") as f:
                f.write(str(self))