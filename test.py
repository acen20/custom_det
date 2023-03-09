from detectron2.engine import DefaultPredictor
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
from detectron2.data import transforms as T
import torch
import cv2
import numpy as np
from model_configs import get_custom_config
import json
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import time
from custom_sahi import get_sahi_detection_model
from trainer import register_data
from tqdm import tqdm
from detectron2.evaluation import COCOEvaluator
import warnings
warnings.filterwarnings("ignore")



def generate_feature_maps(cfg, predictor, base_dir, im_name):
    img = cv2.imread(os.path.join(base_dir, im_name))
    augment_im_size = cfg.INPUT.MIN_SIZE_TEST
    input_img = T.AugInput(img)
    augs = T.AugmentationList([
        T.ResizeShortestEdge(short_edge_length=[augment_im_size, augment_im_size], max_size=1600),
    ])
    image_transformed = input_img.image
    image_transformed = np.moveaxis(image_transformed,-1,0)

    img = torch.tensor(image_transformed, dtype=torch.float)

    output = predictor.model.backbone.bottom_up(img.to(device=cfg.MODEL.DEVICE).unsqueeze(0)/255.0)
    8
    for feat in output.keys():
        output[feat] = output[feat].unsqueeze(4).transpose(1,4).squeeze().detach().cpu()

    plt.figure(figsize=(12,12))
    plt.subplot(3,2,1)
    plt.axis("off")
    _ = plt.imshow(np.moveaxis(image_transformed, 0, -1)[:,:,::-1], cmap="winter_r")

    plt.subplot(3,2,2)
    plt.axis("off")
    _ = plt.imshow(torch.sum(output['res2'], 2), cmap="plasma_r")

    plt.subplot(3,2,3)
    plt.axis("off")
    _ = plt.imshow(torch.sum(output['res3'], 2), cmap="plasma_r")

    plt.subplot(3,2,4)
    plt.axis("off")
    _ = plt.imshow(torch.sum(output['res4'], 2), cmap="plasma_r")

    plt.subplot(3,2,5)
    plt.axis("off")
    _ = plt.imshow(torch.sum(output['res5'], 2), cmap="plasma_r")

    plt.savefig('results/feature_maps.png')


def start_inference(predictor, cfg, test_annotations, base_dir, 
                    images, custom_metadata, use_sahi=True):
    wo_sahi_errors = []
    sahi_errors = []

    if use_sahi:
        ## PREPARE SAHI MODEL
        sahi_predictor = get_sahi_detection_model(cfg, custom_metadata)

    for img in tqdm(images, 'Testing'):
        file_name = img
        img = os.path.join(base_dir, img)
        im = cv2.imread(img)


        ## GET ACTUAL ANNOTATIONS FOR THE IMAGE
        test_im_id = [test_im['id'] for test_im in test_annotations['images'] if test_im['file_name'] == file_name][0]
        actual_annotations = [ann for ann in test_annotations['annotations'] if ann['image_id'] == test_im_id]
        

        ## CREATE VISUALIZER OBJECT
        v = Visualizer(im[:,:,::-1],
                    metadata=custom_metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW
        )


        if use_sahi:
            # PREDICT
            outputs_sahi = sahi_predictor(im[:,:,::-1])
            
            ## VISUALIZE AND SAVE
            out_sahi = v.draw_instance_predictions(outputs_sahi["instances"])
            out_sahi.save(f"results/SAHI_{file_name}")

            ## CALCULATE AND APPEND ERROR TO LIST
            sahi_error = abs(len(outputs_sahi['instances']) - len(actual_annotations))
            sahi_errors.append(sahi_error)

        
        outputs_original = predictor(im)
               
        outputs_original["instances"] = outputs_original["instances"].to('cpu')
        wo_sahi_error = abs(len(outputs_original['instances']) - len(actual_annotations))  
        wo_sahi_errors.append(wo_sahi_error)
        
        out_original = v.draw_instance_predictions(outputs_original["instances"])
        
        v = Visualizer(im[:,:,::-1],
                    metadata=custom_metadata, 
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE_BW
        )
              
        out_original.save(f"results/{file_name}")

    print("Output images saved to results/")
    wo_sahi_mae = np.average(wo_sahi_errors)
    sahi_mae = np.average(sahi_errors)

    print(f"MAE:\t\t {wo_sahi_mae:.2f}")
    print(f"MAE(SAHI):\t {sahi_mae:.2f}")

def evaluate_model():
    print("Evaluating...")
    eval = COCOEvaluator("custom_test", output_dir="./output", allow_cached_coco=True)
    eval.evaluate()


def test_model(train_dir, test_dir):
    base_dir = test_dir
    cfg = get_custom_config()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    predictor = DefaultPredictor(cfg)
    _ , _ , custom_metadata = register_data(train_dir, test_dir)

    formats = ["png","jpg"]

    ## GET IMAGES FROM THE BASE DIRECTORY
    all_files = os.listdir(base_dir)
    images = [file_ for file_ in all_files if file_.split('.')[-1] in formats]

    ## GENERATE FEATURE MAPS FOR AN EXAMPLE IMAGE
    generate_feature_maps(cfg, predictor, base_dir, images[0])

    ## LOAD TEST ANNOTATIONS
    with open(f"{base_dir}/test.json") as f:
        test_annotations = json.load(f)

    start_inference(predictor=predictor,
                    cfg=cfg,
                    test_annotations = test_annotations, 
                    base_dir = base_dir,
                    images = images, 
                    custom_metadata=custom_metadata,
                    use_sahi=False)
    
    evaluate_model()

if __name__ == "__main__":
    TRAIN_DIR = "../Dataset/SPIKE Dataset/positive"
    TEST_DIR = "../Dataset/SPIKE Dataset/testImages_SPIKE"
    test_model(TRAIN_DIR, TEST_DIR)