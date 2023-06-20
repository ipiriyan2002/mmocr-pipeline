from code_gen_lib.code_block import *
import code_gen_lib.mmocr_config_writers.configs.recog_configs as cfg
import os


class TextRecModelConfig:
    DEFAULT_SCHEDULES = None
    DEFAULT_SCHEDULE = None
    DEFAULT_RUNTIME = None

    def __init__(self, dataset, model, base=None, epochs=20,
                 schedule=None, has_val=False,
                 train_batch_size=64, test_batch_size=32):

        self.checkAndAssign(model, base, epochs)
        self.model = model

        if schedule is None:
            self.schedule = f"../_base_/schedules/{self.DEFAULT_SCHEDULE}"
        else:
            assert (
                    schedule in self.DEFAULT_SCHEDULES), f"Available schedules for text detection: {self.DEFAULT_SCHEDULES}"

            self.schedule = f"../_base_/schedules/{self.schedule}"

        self.fname = f"{model}_{epochs}_{dataset}.py"
        self.base_file = base if not(base is None) else cfg.model_dict[model]["base"][0]

        if os.path.exists(dataset):
            self.dataset_path = dataset
            self.dataset = dataset.split("/")[-1]
        else:
            self.dataset_path = f"../_base_/datasets/{dataset}"
            self.dataset = dataset.split(".")[0]

        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs

    def checkAndAssign(self, model, base, epochs):

        model_dict = cfg.model_dict[model]

        self.DEFAULT_RUNTIME = cfg.DEFAULT_RUNTIME
        self.DEFAULT_SCHEDULE = cfg.DEFAULT_SCHEDULE
        self.DEFAULT_SCHEDULES = cfg.DEFAULT_SCHEDULES

        if not(base is None):
            assert (base in model_dict["base"]), f"Available bases for {model} :: {model_dict['base']}, but got {base}"
        assert (isinstance(epochs, int)), "Epochs should be integer"

    def __str__(self, indent=""):

        baseStatement = StatementBlock(statements=["_base_ = [",
                                                   f"\t'{self.base_file}',",
                                                   f"\t'{self.DEFAULT_RUNTIME}',",
                                                   f"\t'{self.dataset_path}',",
                                                   f"\t'{self.schedule}',",
                                                   "\t]"])

        head = f"{self.dataset}_textrecog"
        assigns = [f"{head}_train = _base_.{head}_train", f"{head}_train.pipeline = _base_.train_pipeline",
                   f"{head}_test = _base_.{head}_test", f"{head}_test.pipeline =  _base_.test_pipeline"]

        if self.has_val:
            assigns.extend([f"{head}_val = _base_.{head}_val", f"{head}_val.pipeline = _base_.test_pipeline"])
        assignStatements = StatementBlock(statements=assigns)

        dataloaderStatements = []

        for split in ["train", "test", "val"]:
            if self.has_val or (split in ["train", "test"]):
                isTrain = split == "train"
                batch_size = self.train_batch_size if isTrain else self.test_batch_size
                nw = 8 if isTrain else 4
                assigns = [f"{split}_dataloader = dict(",
                           f"\tbatch_size={batch_size},"
                           f"\tnum_workers={nw},",
                           f"\tpersistent_workers=True,",
                           f"\tsampler=dict(type='DefaultSampler', shuffle={isTrain}),",
                           f"\tdataset={head}_{split})"]

                if split in ["val", "test"]:
                    assigns.insert(4,"\tdrop_last=False,")
            else:
                assigns = ["val_dataloader = test_dataloader"]

            dataloaderStatements.append(StatementBlock(statements=assigns))
            dataloaderStatements.append("")

        evaluatorStatement = StatementBlock(statements=[f"val_evaluator = dict(dataset_prefixes=['{self.dataset}'])","test_evaluator = val_evaluator", ""])
        autoscaleStatement = StatementBlock(statements=[f"auto_scale_lr = dict(base_batch_size={self.train_batch_size} * 4)"])

        finals = [baseStatement, "",
                  assignStatements, "",
                  ]
        finals.extend(dataloaderStatements)
        finals.append(evaluatorStatement)
        finals.append(autoscaleStatement)
        finals.append("")
        finals.append(f"train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={self.epochs}, val_interval=20)")

        finalStatement = StatementBlock(statements=finals)

        return str(finalStatement)

    def __call__(self):

        model_name = self.model
        base_path = f"../configs/textrecog/{model_name}/"
        mmocr_path = f"../mmocr/configs/textrecog/{model_name}/"

        if os.path.exists(base_path):
            save_path = os.path.join(base_path, self.fname)
        elif os.path.exists(mmocr_path):
            save_path = os.path.join(mmocr_path, self.fname)
        else:
            save_path = f"./{self.fname}"

        with open(save_path, "w") as f:
            f.write(str(self))

        return save_path