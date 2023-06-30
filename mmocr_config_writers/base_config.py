import os

class BaseConfig:

    DEFAULT_TASKS = ["textdet", "textrecog"]

    def __init__(self, task, save_path=None):

        assert (task in self.DEFAULT_TASKS), f"Expected one of the following tasks: {self.DEFAULT_TASKS} but received {task}"
        self.task = task

        self.save_path = self.getSavePath(save_path)    

    