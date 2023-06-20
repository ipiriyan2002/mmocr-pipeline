import os

class BaseConfig:

    DEFAULT_TASKS = ["textdet", "textrecog"]

    def __init__(self, dataset_name, task, save_path=None):

        self.dataset = dataset_name

        assert (task in self.DEFAULT_TASKS), f"Expected one of the following tasks: {self.DEFAULT_TASKS} but received {task}"
        self.task = task

        self.save_path = self.getSavePath(save_path)    

    