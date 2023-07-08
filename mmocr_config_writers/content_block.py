import os
from utils.code_block import *

"""
Class to write parameter settings
"""
class ContentBlock:

    def __init__(self,
                 log_interval,
                 checkpoint_interval,
                 optimizer_params,
                 schedulers,
                 cfgs,
                 ):

        assert (isinstance(schedulers, (dict, list))), f"Expected a single dictionary or a list of dictionaries but got {type(schedulers)}"
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval

        self.optimzer_params = optimizer_params
        self.schedulers = schedulers
        self.cfgs = cfgs


    """
    Get hook statements
    """
    def getHooksStatement(self, indent):

        hook_stats = [indent+f"default_hooks = dict(\n"]

        if not(self.log_interval is None):
            hook_stats.append(indent+f"\tlogger=dict(type='LoggerHook', interval={self.log_interval}),")

        if not(self.checkpoint_interval is None):
            hook_stats.append(indent+f"\tcheckpoint=dict(type='CheckpointHook', interval={self.checkpoint_interval}),")
        hook_stats.append(indent+")")

        if len(hook_stats) == 2:
            hook_stats = []

        return StatementBlock(statements=hook_stats)

    """
    Unpack optimizer parameters
    """
    def unpackOptParams(self, indent):

        if self.optimzer_params is None:
            return ""
        
        kvals = self.optimzer_params.items()

        optimizer_statements = indent+"dict("

        for key, value in kvals:

            temp = f"{key}={value},"
            if isinstance(value, (bool, type(None))):
                pass
            elif isinstance(value, str):
                temp = f"{key}='{value}',"


            optimizer_statements += temp

        optimizer_statements += ")"

        optimizerStatBlock = StatementBlock(statements=[
            indent+"optim_wrapper = dict(",
            indent+"\ttype='OptimWrapper',",
            indent+f"\toptimizer={optimizer_statements}",
            indent+")"
        ])

        return optimizerStatBlock

    """
    Unpack the scheduler settings
    """
    def unpackSchedulers(self, indent):

        if self.schedulers is None:
            return ""

        if isinstance(self.schedulers, dict):
            self.schedulers = [self.schedulers]

        results = [indent+"param_scheduler = ["]

        for param_schedule in self.schedulers:

            if param_schedule is None:
                continue

            temp_message = indent+"dict("

            for key, value in param_schedule.items():

                temp = f"{key}={value},"

                if isinstance(value, (bool, type(None))):
                    pass
                elif isinstance(value, str):
                    temp = f"{key}='{value}',"

                temp_message += temp

            temp_message += ")"

            results.append(indent+"\t"+temp_message+",")

        results.append(indent+"]")

        schedulerBlock = StatementBlock(statements=results)

        return schedulerBlock


    """
    Unpack the (train/test/val) cfg file
    """
    def unpackCfgDict(self, dict_):

        result = "dict("

        for key, value in dict_.items():
            temp = f"{key}={value},"

            if isinstance(value, (bool, type(None))):
                pass
            elif isinstance(value, str):
                temp = f"{key}='{value}',"

            result += temp

        result += ")"

        return result

    def unpackCfgs(self, indent):

        if self.cfgs is None:
            return ""

        results = []

        for key,value in self.cfgs.items():

            if not(value is None):

                unpacked_dict = self.unpackCfgDict(value)

                results.append(indent+f"{key}={unpacked_dict}")

        return StatementBlock(statements=results)

    def __str__(self, indent=""):

        final_statement = self.getStatement(indent)

        return str(final_statement)
        
    def getStatement(self, indent):

        #Default hooks
        hookStatement = self.getHooksStatement(indent)

        #Optimizer setup
        optStatement = self.unpackOptParams(indent)

        #Scheduler setup
        schedulerStatement = self.unpackSchedulers(indent)

        #Configs setup
        cfgStatement = self.unpackCfgs(indent)

        final_statement = StatementBlock(statements=[hookStatement, optStatement, schedulerStatement, cfgStatement])

        return final_statement