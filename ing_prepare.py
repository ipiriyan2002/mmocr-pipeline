
"""
Currently Ing Dataset is Not working properly
"""


dataset_dict = dict(
    type="ing",
    init_params= dict(name="ing", tasks = ["det"], save_dir="../Ingredients/data/", generator=None),
    prepare_params=dict(train=dict(img_paths=["../Ingredients/Batch3/Original_Image/correct"], ann_paths=["../Ingredients/Batch3/f2150000.csv"], split="train"),
                        test=dict(img_paths=["../Ingredients/Batch1/correct"], ann_paths=["../Ingredients/Batch1/Appen_info/f2143224.csv"], split="test"),
                        val=None)
)


#Include evaluation writing later
det_model_dict = dict(
    train_datasets=["ing.py"],
    val_datasets=None,
    test_datasets=["ing.py"],
    model="dbnet",
    backbone="resnet18",
    neck="fpnc",
    base=None,
    epochs=40,
    schedule=None,
    has_val=True,
    train_batch_size=16,
    test_batch_size=1,
    contents = dict(
        log_interval=1,
        checkpoint_interval=1,
        optimizer_params=dict(
            type="SGD", lr=0.007, momentum=0.9, weight_decay=0.0001
        ),
        schedulers=[
            dict(type="ConstantLR", factor=1.0)
        ],
        cfgs=dict(
            train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=40, val_interval=1),
            val_cfg=None,
            test_cfg=None
        ),
    )
)

recog_model_dict = None
