

dataset_dict = dict(
    type="cordv2",
    init_params= dict(name="cordv2", tasks = ["det", "recog"], save_dir=None),
    prepare_params=dict(train=dict(img_paths=None, ann_paths=None, split="train"),
                        test=dict(img_paths=None, ann_paths=None, split="test"),
                        val=dict(img_paths=None, ann_paths=None, split="val"))
)


#Unpack the below dictionary into textdet and textrecog dict
det_model_dict=dict(
    train_datasets="cordv2.py",
    val_datasets="cordv2.py",
    test_datasets="cordv2.py",
    model="dbnet",
    backbone="resnet18",
    neck="fpnc",
    base=None,
    epochs=40,
    schedule=None,
    has_val=True,
    train_batch_size=16,
    test_batch_size=1,
    contents=dict(
        log_interval=None,
        checkpoint_interval=None,
        optimizer_params=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001),
        schedulers=[dict(type='ConstantLR', factor=1.0),],
        cfgs=dict(train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=2),
                  val_cfg=None,
                  test_cfg=None)
    )
)

recog_model_dict = None