from utils.code_block import *
import mmocr_config_writers.configs.det_configs as cfg
from mmocr_config_writers.content_block import ContentBlock
import os


class TextDetModelConfig:
    DEFAULT_SCHEDULES = None
    DEFAULT_SCHEDULE = None
    DEFAULT_RUNTIME = None

    def __init__(self, train_datasets, val_datasets, test_datasets, model, backbone, neck, epochs,
                 schedule=None, has_val=False,
                 train_batch_size=16, test_batch_size=1, contents=None):

        self.checkAndAssign(model, backbone, neck, epochs)
        self.model = model
        self.contents=contents

        if schedule is None:
            self.schedule = f"../_base_/schedules/{self.DEFAULT_SCHEDULE}"
        else:
            assert (
                    schedule in self.DEFAULT_SCHEDULES), f"Available schedules for text detection: {self.DEFAULT_SCHEDULES}"

            self.schedule = f"../_base_/schedules/{self.schedule}"

        self.dataset_name, self.dataset_base_paths, self.datasets = self.gatherDatasets(train_datasets, val_datasets, test_datasets)
        self.fname = f"{model}_{backbone}_{neck}_{epochs}_{self.dataset_name}.py"
        self.base_file = f"_base_{model}_{backbone}_{neck}.py"

        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs


    def cleanDatasetNames(self, dataset):

        if isinstance(dataset, str):
            datasets = [dataset]
        elif dataset is None:
            return []
        else:
            datasets = dataset

        cleaned = []

        for ds in datasets:
            if os.path.exists(ds):
                cleaned.append(ds)
            else:
                cleaned.append(ds.split(".")[0]+".py")

        return cleaned

    def gatherDatasets(self, train, val, test):
        assert isinstance(train, (str, list)), "Expect training datasets to be either a single dataset or multiple datasets"
        assert isinstance(val, (str, list, type(None))), "Expect testing datasets to be either a single dataset or multiple datasets or None"
        assert isinstance(test, (str, list, type(None))), "Expect testing datasets to be either a single dataset or multiple datasets or None"

        train = self.cleanDatasetNames(train)
        val = self.cleanDatasetNames(val)
        test = self.cleanDatasetNames(test)

        dataset_name = train[0].split("/")[-1].split(".")[0] if len(train) == 1 else "multi"

        all_datasets = []
        all_datasets.extend(train)
        all_datasets.extend(val)
        all_datasets.extend(test)

        dataset_paths = []

        for ds_split in [train, val, test]:

            for ds in ds_split:

                if os.path.exists(ds):
                    dataset_paths.append(ds)
                else:
                    dataset_paths.append(f"../_base_/datasets/{ds}")

        datasets = dict(train=train, val=val, test=test)

        return dataset_name, list(set(dataset_paths)), datasets
            




    def checkAndAssign(self, model, backbone, neck, epochs):

        model_dict = cfg.model_dict[model]

        self.DEFAULT_RUNTIME = cfg.DEFAULT_RUNTIME
        self.DEFAULT_SCHEDULE = cfg.DEFAULT_SCHEDULE
        self.DEFAULT_SCHEDULES = cfg.DEFAULT_SCHEDULES

        assert (backbone in model_dict[
            "backbones"]), f"Avaiable backbones for {model} :: {model_dict['backbones']}, but got {backbone}"
        assert (neck in model_dict["necks"]), f"Avaiable necks for {model} :: {model_dict['necks']}, but got {neck}"
        assert (isinstance(epochs, int)), "Epochs should be integer"

    def getBaseStatement(self, indent):
        bases = ["_base_ = [",
                 f"\t'{self.base_file}',"
                 f"\t'{self.DEFAULT_RUNTIME}',",
                 f"\t'{self.schedule}',",
                 ]

        for ds_path in self.dataset_base_paths:
            bases.append(f"\t'{ds_path}',")

        bases.append("\t]")

        baseStatement = StatementBlock(statements=bases)

        return baseStatement


    def getAssignStatement(self, head, indent):

        lists = []
        assigns = []

        for split, split_vals in self.datasets.items():
            pipeline_split = split if split in ["train", "test"] else "test"
            if len(split_vals) == 1:
                data_head = split_vals[0].split("/")[-1].split(".")[-2]
                assigns.extend([f"{head}_{split} = _base_.{data_head}_textdet_{split}", f"{head}_{split}.pipeline = _base_.{pipeline_split}_pipeline"])
            else:
                list_message = [f"{split}_list = ["]
                for split_val in split_vals:
                    data_head = split_vals[0].split("/")[-1].split(".")[-2]
                    list_message.append(f"\t_base_.{data_head}_textdet_{split},")

                list_message.append("]")
                list_message.append("")
                lists.append(StatementBlock(statements=list_message))

                assign_message = f"{head}_{split} = dict(type='ConcatDataset', datasets={split}_list, pipeline=_base_.{pipeline_split}_pipeline)"

                assigns.extend([assign_message,""])

        results = []
        results.extend(lists)
        results.extend(assigns)
        assign_statement = StatementBlock(statements=results)

        return assign_statement

    def __str__(self, indent=""):

        baseStatement = self.getBaseStatement(indent)

        head = f"{self.dataset_name}_textdet"
        assignStatements = self.getAssignStatement(head, indent)

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
            else:
                assigns = ["val_dataloader = test_dataloader"]

            dataloaderStatements.append(StatementBlock(statements=assigns))
            dataloaderStatements.append("")

        autoscaleStatement = StatementBlock(statements=[f"auto_scale_lr = dict(base_batch_size={self.train_batch_size})"])

        finals = [baseStatement, "",
                  assignStatements, "",
                  ]
        finals.extend(dataloaderStatements)
        finals.append(autoscaleStatement)

        if not(self.contents is None):
            contentStatement = ContentBlock(**self.contents)
            finals.append(contentStatement.getStatement(indent))

        finalStatement = StatementBlock(statements=finals)

        return str(finalStatement)

    def __call__(self):

        model_name = self.model if (self.model != "mask-rcnn") else "maskrcnn"
        base_path = f"../configs/textdet/{model_name}/"
        mmocr_path = f"../mmocr/configs/textdet/{model_name}/"

        if os.path.exists(base_path):
            save_path = os.path.join(base_path, self.fname)
        elif os.path.exists(mmocr_path):
            save_path = os.path.join(mmocr_path, self.fname)
        else:
            save_path = f"./{self.fname}"

        with open(save_path, "w") as f:
            f.write(str(self))

        return save_path
