from utils.code_block import *
import mmocr_config_writers.configs.recog_configs as cfg
from mmocr_config_writers.content_block import ContentBlock
import os

"""
Recognition model configuration setup
"""
class TextRecModelConfig:
    DEFAULT_SCHEDULES = None
    DEFAULT_SCHEDULE = None
    DEFAULT_RUNTIME = None

    def __init__(self, train_datasets, val_datasets, test_datasets, model, base=None, vocab=None, epochs=20,
                 schedule=None, has_val=False,
                 train_batch_size=64, test_batch_size=32, contents=None):

        #Make sure models, bases, vocab and epochs are valid
        vocab = self.checkAndAssign(model, base, vocab, epochs)
        self.model = model
        self.contents = contents

        #Get path to schedule
        if schedule is None:
            self.schedule = f"../_base_/schedules/{self.DEFAULT_SCHEDULE}"
        else:
            assert (
                    schedule in self.DEFAULT_SCHEDULES), f"Available schedules for text detection: {self.DEFAULT_SCHEDULES}"

            self.schedule = f"../_base_/schedules/{self.schedule}"

        #Gather datasets
        self.dataset_name, self.dataset_base_paths, self.datasets = self.gatherDatasets(train_datasets, val_datasets,
                                                                                        test_datasets)
        self.fname = f"{model}_{epochs}_{self.dataset_name}.py"
        self.base_file = base if not (base is None) else cfg.model_dict[model]["base"][0]
        self.vocab = vocab if not (vocab is None) else cfg.DEFAULT_VOCAB

        self.has_val = has_val
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs

    """
    Clean dataset names such that they are either valid paths or just file names
    """
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
                cleaned.append(ds.split(".")[0] + ".py")

        return cleaned

    """
    Gather datasets, by doing pre-processing including cleaning names and formatting MMOCR file paths
    """
    def gatherDatasets(self, train, val, test):
        assert isinstance(train,
                          (str, list)), "Expect training datasets to be either a single dataset or multiple datasets"
        assert isinstance(val,
                          (str, list, type(None))), "Expect validation datasets to be either a single dataset or multiple datasets"
        assert isinstance(test,
                          (str, list, type(None))), "Expect testing datasets to be either a single dataset or multiple datasets"

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

        datasets = dict(train=train, test=test, val=val)

        return dataset_name, list(set(dataset_paths)), datasets

    """
    Save vocabulary if it does not exist if not get file path to vocabulary
    """
    def saveVocab(self, vocab):

        vocab_file_save_name = f"custom_{self.dataset_name}.txt"

        if os.path.exists("../mmocr/dicts/"):
            save_path = os.path.join("../mmocr/dicts/", vocab_file_save_name)
            ref_save_path =f"{{ fileDirname }}/../../../dicts/{vocab_file_save_name}"
        elif os.path.exists("../dicts/"):
            save_path = os.path.join("../dicts/", vocab_file_save_name)
            ref_save_path =f"{{ fileDirname }}/../../../dicts/{vocab_file_save_name}"
        else:
            save_path = f"./{vocab_file_save_name}"
            ref_save_path =  os.path.join(os.getcwd(), vocab_file_save_name)

        with open(save_path, "w") as f:

            f.write("\n".join(vocab))

        return ref_save_path


    """
    Make sure models, bases, vocabs and epochs are valid
    """
    def checkAndAssign(self, model, base, vocab, epochs):

        model_dict = cfg.model_dict[model]


        #Load runtimes and schedules
        self.DEFAULT_RUNTIME = cfg.DEFAULT_RUNTIME
        self.DEFAULT_SCHEDULE = cfg.DEFAULT_SCHEDULE
        self.DEFAULT_SCHEDULES = cfg.DEFAULT_SCHEDULES

        if not (base is None):
            assert (base in model_dict["base"]), f"Available bases for {model} :: {model_dict['base']}, but got {base}"

        if not (vocab is None) and isinstance(vocab, list):
            vocab = self.saveVocab(vocab)
        elif not (vocab is None) and not(os.path.exists(vocab)):
            assert (vocab in cfg.vocabs), f"Available vocabularies include {cfg.vocabs}, but got {vocab}.\n" \
                                          f"Provide from available vocabularies or a valid vocab txt file"

            vocab = f"{{ fileDirname }}/../../../dicts/{vocab}"

        assert (isinstance(epochs, int)), "Epochs should be integer"

        return vocab

    """
    Get Vocab statements
    """
    def getVocabStatement(self, indent):

        vocabs = [
            indent + "_base_.model.decoder.dictionary.update(",
            indent + f"\tdict(dict_file={self.vocab}, with_unknown=True, unknown_token=None)",
            indent + ")"
        ]

        vocabStatement = StatementBlock(statements=vocabs)

        return vocabStatement


    def getBaseStatement(self, indent):
        bases = [indent + "_base_ = [",
                 indent + f"\t'{self.base_file}',",
                 indent + f"\t'{self.DEFAULT_RUNTIME}',",
                 indent + f"\t'{self.schedule}',",
                 ]

        for ds_path in self.dataset_base_paths:
            bases.append(indent + f"\t'{ds_path}',")

        bases.append(indent + "]")

        baseStatement = StatementBlock(statements=bases)

        return baseStatement

    def getAssignStatement(self, head, indent):

        lists = []
        assigns = []

        for split, split_vals in self.datasets.items():
            pipeline_split = split if split in ["train", "test"] else "test"
            if split_vals == None or split_vals == []:
                if split == "val":
                    assigns.extend([
                        indent + f"{head}_val = {head}_test"
                    ])
            elif len(split_vals) == 1:
                data_head = split_vals[0].split("/")[-1].split(".")[-2]
                assigns.extend([indent + f"{head}_{split} = _base_.{data_head}_textrecog_{split}",
                                indent + f"{head}_{split}.pipeline = _base_.{pipeline_split}_pipeline"])
            else:
                list_message = [indent + f"{split}_list = ["]
                for split_val in split_vals:
                    data_head = split_val.split("/")[-1].split(".")[-2]
                    list_message.append(indent + f"\t_base_.{data_head}_textrecog_{split},")

                list_message.append(indent + "]")
                list_message.append("")
                lists.append(StatementBlock(statements=list_message))

                assign_message = indent + f"{head}_{split} = dict(type='ConcatDataset', datasets={split}_list, pipeline=_base_.{pipeline_split}_pipeline)"

                assigns.extend([assign_message, ""])

        results = []
        results.extend(lists)
        results.extend(assigns)
        assign_statement = StatementBlock(statements=results)

        return assign_statement

    """
    Get prefixes for each evaluator
    """
    def getPrefixes(self):
        val_prefixes = []
        test_prefixes = []
        for split, split_vals in self.datasets.items():
            if split == "train":
                continue

            pref2add = [split_val.split("/")[-1].split(".")[-2] for split_val in split_vals]

            if split == "val":
                val_prefixes.extend(pref2add)
            elif split == "test":
                test_prefixes.extend(pref2add)

        return val_prefixes, test_prefixes

    def __str__(self, indent=""):

        baseStatement = self.getBaseStatement(indent)

        vocabStatement = self.getVocabStatement(indent)

        head = f"{self.dataset_name}_textrecog"
        assignStatements = self.getAssignStatement(head, indent)

        dataloaderStatements = []

        val_prefixes, test_prefixes = self.getPrefixes()
        eval_statements = []
        for split in ["train", "test", "val"]:
            if self.has_val or (split in ["train", "test"]):
                isTrain = split == "train"
                batch_size = self.train_batch_size if isTrain else self.test_batch_size
                nw = 8 if isTrain else 4
                assigns = [indent + f"{split}_dataloader = dict(",
                           indent + f"\tbatch_size={batch_size},",
                           indent + f"\tnum_workers={nw},",
                           indent + f"\tpersistent_workers=True,",
                           indent + f"\tsampler=dict(type='DefaultSampler', shuffle={isTrain}),",
                           indent + f"\tdataset={head}_{split})"]

                if split in ["val", "test"]:
                    assigns.insert(4, indent + "\tdrop_last=False,")

                    prefixes = val_prefixes if split == "val" else test_prefixes
                    eval_statements.append(indent + f"{split}_evaluator = dict(dataset_prefixes={prefixes})")


            else:
                assigns = ["val_dataloader = test_dataloader"]
                eval_statements.append("val_evaluator = test_evaluator")

            dataloaderStatements.append(StatementBlock(statements=assigns))
            dataloaderStatements.append("")


        evaluatorStatement = StatementBlock(
            statements=eval_statements)
        autoscaleStatement = StatementBlock(
            statements=[indent + f"auto_scale_lr = dict(base_batch_size={self.train_batch_size} * 4)"])

        finals = [baseStatement, "",
                  vocabStatement, "",
                  assignStatements, "",
                  ]
        finals.extend(dataloaderStatements)
        finals.append(evaluatorStatement)
        finals.append(autoscaleStatement)

        if not (self.contents is None):
            contentStatement = ContentBlock(**self.contents)
            finals.append(contentStatement.getStatement(indent))

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

        if ".py" in save_path:
            with open(save_path, "w", encoding='utf-8') as f:
                f.write(str(self))
        else:
            with open(save_path, "w") as f:
                f.write(str(self))

        return save_path

#%%
