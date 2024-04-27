import torch
import argparse
import cv2
import detect_utils
import numpy as np
from PIL import Image
from model import get_model

## If you have CUDA ERRORS make sure to install cuda116

# Construct the argument parser.
parser = argparse.ArgumentParser()

parser.add_argument(
    '-i', '--input', default='input/image_1.jpg',
    help='path to input input image'
)
parser.add_argument(
    '-t', '--threshold', default=0.5, type=float,
    help='detection threshold'
)
parser.add_argument(
    '-m', '--model', default='v2',
    help='faster rcnn resnet50 fpn or fpn v2',
    choices=['v1', 'v2']
)

args = vars(parser.parse_args())
# args = {'input': '.\\..\\dataset\\RBK_TDT17\\1_train-val_1min_aalesund_from_start\\img1\\000003.jpg', 'threshold': 0.5, 'model': 'v2'}
print("Arguments: ", args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device, args['model'])

image = Image.open(args['input']).convert('RGB')
image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# detect outputs.
with torch.no_grad():
    boxes, classes, labels, class_count = detect_utils.predict(image, model, device, args['threshold'])

# draw bounding boxes.
image = detect_utils.draw_boxes(boxes, classes, labels, image_bgr, classes_count=class_count)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_t{''.join(str(args['threshold']).split('.'))}_{args['model']}"
cv2.imshow('Image', image)
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)

