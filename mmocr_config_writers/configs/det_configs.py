model_dict = dict(
    dbnet=dict(backbones=["resnet18", "resnet50-dcnv2"], necks=["fpnc"]),
    dbnetpp=dict(backbones=["resnet50-dcnv2"], necks=["fpnc"]),
    drrg=dict(backbones=["resnet50"], necks=["fpn-unet"]),
    fcenet=dict(backbones=["resnet50"], necks=["fpn", "dcnv2_fpn"]),
    panet=dict(backbones=["resnet18","resnet50"], necks=["fpem-ffm"]),
    psenet=dict(backbones=["resnet50"], necks=["fpnf"]),
    textsnake=dict(backbones=["resnet50"], necks=["fpn-unet"]),
)
model_dict["mask-rcnn"] = dict(backbones=["resnet50"], necks=["fpn"])

#SCHEDULES
DEFAULT_SCHEDULES = ["schedule_adam_600e.py", "schedule_sgd_100k.py", "schedule_sgd_1200e.py"]
DEFAULT_SCHEDULE = "schedule_sgd_1200e.py"
DEFAULT_RUNTIME = '../_base_/default_runtime.py'