
#Currently using base files available in mmocr
model_dict = dict(
    abinet=dict(base=["_base_abinet.py", "_base_abinet-vision.py"]),
    aster=dict(base=["_base_aster.py"]),
    crnn=dict(base=["_base_crnn_mini-vgg.py"]),
    master=dict(base=["_base_master_resnet31.py"]),
    nrtr=dict(base=["_base_nrtr_resnet31.py","_base_nrtr_modality-transform.py"]),
    robust_scanner=dict(base=["_base_robustscanner_resnet31.py"]),
    sar=dict(base=["_base_sar_resnet31_parallel-decoder.py"]),
    satrn=dict(base=["_base_satrn_shallow.py"]),
    svtr=dict(base=["_base_svtr-tiny.py"])
)

#SCHEDULES
DEFAULT_SCHEDULES = ["schedule_adadelta_5e.py", "schedule_adam_step_5e.py", "schedule_adamw_cos_6e.py"]
DEFAULT_SCHEDULE = "schedule_adadelta_5e.py"
DEFAULT_RUNTIME = '../_base_/default_runtime.py'


#Vocabulary dicts

vocabs = [
    "chinese_english_digits.txt", "english_digits_symbols.txt", "english_digits_symbols_space.txt",
    "lower_english_digits.txt", "lower_english_digits_space.txt", "sdmgr_dict.txt"
]

DEFAULT_VOCAB = "english_digits_symbols_space.txt"