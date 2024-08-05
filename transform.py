import os
import json
from PIL import Image
import numpy as np
from pycocotools import mask 
from skimage import measure
from itertools import groupby
import cv2

CATEGORIES: dict[str, int] = {
    "SA": 1,
    "LI": 2,
    "RI": 3,
}

def _shift(category_id: int, fragment_id: int) -> int:
    return 10 * (category_id - 1) + fragment_id

def load_masks(path) -> tuple[np.ndarray, list[int], list[int]]:
    seg = np.array(Image.open(path))
    return seg_to_masks(seg)

def seg_to_masks(seg: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    """Convert a binary-encoded multi-label segmentation to masks."""
    category_ids = []
    fragment_ids = []
    masks = []
    for category_id in CATEGORIES.values():
        for fragment_id in range(1, 11):
            mask = np.right_shift(seg, _shift(category_id, fragment_id)) & 1
            if mask.sum() > 0:
                masks.append(mask.astype('uint8'))
                category_ids.append(category_id)
                fragment_ids.append(fragment_id)

    return np.array(masks), category_ids, fragment_ids

def binary_mask_to_polygon(binary_mask):
    # binary_mask = binary_mask >
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.size >= 6:
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    return polygons

def create_coco_annotation(img_id, annotation_id, category_id, binary_mask, image_size):
    fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    # contours = measure.find_contours(binary_mask, 0.5)


    annotation = {
            "segmentation":binary_mask_to_polygon(binary_mask),
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": category_id,
            "id": annotation_id
        }
    return annotation

def convert_to_coco_format(img_dir, ann_dir, output_file):
    coco_dataset = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # category_id = 1
    categories = [{"id": 1, "name": "SA"} , {'id' : 2 , 'name': 'LI'} , {'id':3 , 'name': 'RI'}]
    coco_dataset["categories"] = categories

    annotation_id = 1
    img_id = 1
    files = os.listdir(img_dir)
    files.sort()
    # files = files[:50]
    for img_filename in files:
        img_path = os.path.join(img_dir, img_filename)
        img = Image.open(img_path)
        width, height = img.size
        
        image_info = {
            "file_name": img_filename,
            "height": height,
            "width": width,
            "id": img_id
        }
        coco_dataset["images"].append(image_info)
                
        binary_masks , category_ids , fragment_ids = load_masks(os.path.join(ann_dir, img_filename) )

        for binary_mask , category_id ,fragment_id in zip(binary_masks , category_ids , fragment_ids):
            annotation = create_coco_annotation(img_id, annotation_id, category_id, binary_mask, (width, height))
            # print(annotation)
            coco_dataset["annotations"].append(annotation)
            annotation_id += 1
        img_id += 1
        print(img_id)

    with open(output_file, 'w') as f:
        json.dump(coco_dataset, f, indent=4)

# Paths to your image and annotation directories
img_dir = '/scratch/dr/y.nawar/pengwin/train/input/images/x-ray/'
ann_dir = '/scratch/dr/y.nawar/pengwin/train/output/images/x-ray/'
output_file = '/scratch/dr/y.nawar/pengwin/train/coco_annotations.json'

convert_to_coco_format(img_dir, ann_dir, output_file)
