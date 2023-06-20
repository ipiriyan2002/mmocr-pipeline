
dataset_dict = dict(
    type="ing",
    init_params= dict(name="ing", tasks = ["det", "recog"], save_dir=None),
    prepare_params=dict(train=dict(csv_path=None, img_path=None),
                        test=dict(csv_path=None, img_path=None),
                        val=None)
)

model_dict = dict(
    textdet=dict(
        model="dbnet",
        backbone="resnet18",
        neck="fpnc",
        base=None,
        epochs=200,
        schedule=None,
        has_val=False,
        train_batch_size=16,
        test_batch_size=1
    ),
    textrecog=None
)