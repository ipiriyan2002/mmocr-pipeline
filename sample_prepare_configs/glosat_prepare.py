dataset_dict = dict(
    type="glosat",
    config_save_dir=None,
    init_params= dict(name="glosat", tasks = ["det", "recog"], save_dir=None, generator=None),
    prepare_params=dict(train=dict(img_paths=["C:/Users/ipiri/OneDrive/Desktop/Part 3 Project/glosat/Image/"], ann_paths=["C:/Users/ipiri/OneDrive/Desktop/Part 3 Project/glosat/annotation_info/Batch1/"], split="train"),
                        test=dict(img_paths=["C:/Users/ipiri/OneDrive/Desktop/Part 3 Project/glosat/Image/"], ann_paths=["C:/Users/ipiri/OneDrive/Desktop/Part 3 Project/glosat/annotation_info/Batch2/"], split="test"),
                        val=None)
)


det_model_dict = dict(
    train_datasets=["glosat.py"],
    val_datasets=None,
    test_datasets=["glosat.py"],
    model="dbnet",
    backbone="resnet18",
    neck="fpnc",
    base=None,
    epochs=40,
    schedule=None,
    has_val=False,
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

recog_model_dict = dict(
    train_datasets=["glosat.py"],
    val_datasets=None,
    test_datasets=["glosat.py"],
    model="abinet",
    backbone=None,
    neck=None,
    base="_base_abinet-vision.py",
    epochs=10,
    schedule=None,
    has_val=False,
    train_batch_size=64,
    test_batch_size=32,
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
            train_cfg=dict(type="EpochBasedTrainLoop", max_epochs=10, val_interval=1),
            val_cfg=None,
            test_cfg=None
        ),
    )
)