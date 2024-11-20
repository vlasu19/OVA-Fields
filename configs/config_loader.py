import os
from pathlib import Path
from detectron2.config import get_cfg
from detic.config import add_detic_config
from centernet.config import add_centernet_config

DETIC_PATH = os.environ.get("DETIC_PATH", Path(__file__).parent / "../Detic")

def load_cfg():
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(
        f"{DETIC_PATH}/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
    )
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = (
        False  # For better visualization purpose. Set to False for all classes.
    )
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = (
        f"{DETIC_PATH}/datasets/metadata/lvis_v1_train_cat_info.json"
    )
    return cfg