from detectron2 import model_zoo
from detectron2.config import get_cfg
import torch


def get_custom_config():
    cfg = get_cfg()
    model_to_use = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

    cfg.merge_from_file(model_zoo.get_config_file(model_to_use))

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.5

    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.RPN.IN_FEATURES.insert(0, "p2")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ANCHOR_GENERATOR.SIZES.insert(0, [16])
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_to_use)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.BACKBONE.FREEZE_AT = 5
    cfg.MODEL.RESNETS.DEPTH = 101

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2500

    cfg.INPUT.MIN_SIZE_TRAIN = 768
    cfg.INPUT.MAX_SIZE_TRAIN = 1600
    cfg.INPUT.MIN_SIZE_TEST = 768
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = [0.9, 0.9]
    cfg.INPUT.MASK_FORMAT = "polygon"
    cfg.MODEL.MASK_ON = False
    cfg.RESULTS_DIR = "results"


    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005

    cfg.TEST.DETECTIONS_PER_IMAGE = 1000

    return cfg
