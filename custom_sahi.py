from detectron2.engine import DefaultPredictor
from sahi.models.detectron2 import Detectron2DetectionModel
from sahi.predict import get_sliced_prediction
import torch
from detectron2.structures import Instances, Boxes


def get_sahi_detection_model(original_cfg, custom_metadata, class_remapping=False,
                             confidence=None):
    
    sahi_cfg = original_cfg.clone()
    
    det_model = Detectron2DetectionModel(
        device= sahi_cfg.MODEL.DEVICE,
        mask_threshold=confidence,
        confidence_threshold=confidence,
        load_at_init= False
    )

    sahi_predictor = DefaultPredictor(sahi_cfg)

    det_model.model = sahi_predictor
    keys = {}
    offset = 0
    if class_remapping:
        offset = 1
    for i in range(sahi_cfg.MODEL.ROI_HEADS.NUM_CLASSES):
        keys[str(i+offset)] = custom_metadata.thing_classes[i]
    det_model.category_mapping = keys
    return det_model


class SAHIPredictor(torch.nn.Module):
    def __init__(self, detection_model, min_slice = 480, overlap_ratio=0.2, process_type='GREEDYNMM'):
        super(SAHIPredictor, self).__init__()
        self.detection_model = detection_model
        self.min_slice = min_slice
        self.overlap_ratio = overlap_ratio
        self.process_type = process_type
        
    def sahi_to_detectron_instances(self, image, sahi_annotations):
        im_height = image.shape[0]
        im_width = image.shape[1]
        pred_boxes = []
        scores = []
        pred_classes = []
        #pred_masks = []

        for ann in sahi_annotations:
            pred_boxes.append(ann.bbox.to_xyxy())
            scores.append(ann.score.value)
            pred_classes.append(ann.category.id)
            #pred_masks.append(ann.mask.bool_mask)
            
        detectron_dict = {
            'pred_boxes': Boxes(torch.tensor(pred_boxes)),
            'scores':torch.tensor(scores),
            'pred_classes':torch.tensor(pred_classes),
           # 'pred_masks': torch.tensor(pred_masks)
        }
        
        instances = Instances([im_height, im_width])
        for k in detectron_dict.keys():
            instances.set(k,detectron_dict[k])
        instances = {
            'instances':instances
        }
        return instances


        
    def __call__(self,image):
        result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height = image.shape[0]//2,
            slice_width = image.shape[1]//2,
            overlap_height_ratio = self.overlap_ratio,
            overlap_width_ratio = self.overlap_ratio,
            postprocess_type = self.process_type,
            postprocess_match_metric = "IOS",
            verbose = 2
        )
        
        result = self.sahi_to_detectron_instances(image, result.object_prediction_list)
        return result