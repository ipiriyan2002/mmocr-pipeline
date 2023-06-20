
"""
dataset_dict = dict(
    type="ing",
    init_params= dict(name="ing", tasks = ["det", "recog"], save_dir=None),
    prepare_params=dict(train=dict(csv_path=None, img_path=None),
                        test=dict(csv_path=None, img_path=None),
                        val=None)
)
"""
dataset_dict = dict(
    type="cordv2",
    init_params= dict(name="cordv2", tasks = ["det", "recog"], save_dir=None),
    prepare_params=dict(train=dict(split="train"),
                        test=dict(split="test"),
                        val=dict(split="val"))
)

model_dict = dict(
    textdet=dict(
        dataset="cordv2.py",
        model="dbnet",
        backbone="resnet18",
        neck="fpnc",
        base=None,
        epochs=40,
        schedule=None,
        has_val=True,
        train_batch_size=16,
        test_batch_size=1
    ),
    textrecog=None
)