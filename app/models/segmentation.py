import torch, urllib
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torchvision.utils import draw_segmentation_masks
from torchvision.ops import nms
from PIL import Image, ImageDraw
from app.utils.preprocessing import preprocess_image_maskrcnn
import os, config
import io
import base64

url = "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-paper.txt"

with urllib.request.urlopen(url) as response:
    COCO_CLASS_NAMES = [line.decode('utf-8').strip() for line in response.readlines()]

def load_maskrcnn_model(num_classes=len(COCO_CLASS_NAMES)):
    model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn_v2(pretrained=True)
    torch.save(model.state_dict(), config.MASKRCNN_MODEL_PATH)
    return model

def draw_predictions(image_tensor, pred_boxes, pred_labels, pred_scores, pred_masks, pred_color="blue", true_color="green"):
    tensor = image_tensor.clone().detach().cpu()

    if pred_masks is not None:
        masks = pred_masks.squeeze(1)

        colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
                   np.random.randint(0, 255))
                 for _ in range(len(masks))]

        temp_img = (tensor * 255).to(torch.uint8)
        for mask, color in zip(masks, colors):
            temp_img = draw_segmentation_masks(
                temp_img,
                mask[None],
                alpha=0.8,
                colors=color
            )
        tensor = temp_img.float() / 255

    vis_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    result_image = Image.fromarray(vis_np)
    draw = ImageDraw.Draw(result_image)

    return result_image

def serialize(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return image_base64

def predict_segmentation(image_file, model):
    image_tensor = preprocess_image_maskrcnn(image_file)

    with torch.no_grad():
        outputs = model([image_tensor])
        pred_boxes = outputs[0]['boxes']
        pred_labels = outputs[0]['labels']
        pred_scores = outputs[0]['scores']
        pred_masks = outputs[0]["masks"]

        keep_indices = pred_scores > 0.5
        pred_boxes = pred_boxes[keep_indices]
        pred_labels = pred_labels[keep_indices]
        pred_scores = pred_scores[keep_indices]
        pred_masks = pred_masks[keep_indices]
        nms_indices = nms(pred_boxes, pred_scores, 0.5)
        pred_boxes = pred_boxes[nms_indices].cpu().numpy()
        pred_labels = pred_labels[nms_indices].cpu().numpy()
        pred_scores = pred_scores[nms_indices].cpu().numpy()
        pred_masks = pred_masks[nms_indices]
        binary_masks = pred_masks > 0.5
        output_image = draw_predictions(image_tensor, pred_boxes, pred_labels, pred_scores, binary_masks)
        output_image = serialize(output_image)

    return {"predictions": [output_image]}

