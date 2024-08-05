from glob import glob
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure
from sklearn.utils import resample
from peng_utils import load_masks , masks_to_seg
import json
import os
from pycocotools.mask import decode
from PIL import Image


def json_to_tif(json_path, output_path, threshold=0.5):
    f_path = os.path.join(json_path)
    print(f_path)
    with open(f_path, 'r') as file:
        data = json.load(file)

    # Filter the predictions based on the scores
    filtered_indices = [i for i, score in enumerate(data['scores']) if score >= threshold]

    data = {
        "labels": [data['labels'][i] for i in filtered_indices],
        "scores": [data['scores'][i] for i in filtered_indices],
        "bboxes": [data['bboxes'][i] for i in filtered_indices],
        "masks": [data['masks'][i] for i in filtered_indices]
    }
    mask_rle = data['masks']
    mask = decode(mask_rle)
    frag_ids = []
    sa = 1
    li = 1
    ri = 1
    for i in data['labels']:
        if i == 1:
            frag_ids.append(sa)
            sa += 1
        elif i== 2:
            frag_ids.append(li)
            li += 1
        else:
            frag_ids.append(ri)
            ri += 1
    mask = mask.transpose(2,0,1).astype('uint8')
    cats = []
    for cat in data['labels']:
        cats.append(cat + 1)

    seg = masks_to_seg(mask, cats, frag_ids)
    # print(np.unique(seg))
    # masks , _ , _ = seg_to_masks(seg)
    # print(np.unique(masks[2]))
    Image.fromarray(seg).save(output_path)
    return seg
# def json_to_tif(data, output_path = None, threshold=0.5):
#     # f_path = os.path.join(json_path)
#     # print(f_path)
#     # with open(f_path, 'r') as file:
#     #     data = json.load(file)

#     # Filter the predictions based on the scores
#     if threshold > max(data['scores']):
#         threshold = min(data['scores'])

#     filtered_indices = [i for i, score in enumerate(data['scores']) if score >= threshold]

#     data = {
#         "labels": [data['labels'][i] for i in filtered_indices],
#         "scores": [data['scores'][i] for i in filtered_indices],
#         "bboxes": [data['bboxes'][i] for i in filtered_indices],
#         "masks": [data['masks'][i] for i in filtered_indices]
#     }
#     mask_rle = np.array(data['masks'])
#     frag_ids = []
#     sa = 1
#     li = 1
#     ri = 1
#     for i in data['labels']:
#         if i == 1:
#             frag_ids.append(sa)
#             sa += 1
#         elif i== 2:
#             frag_ids.append(li)
#             li += 1
#         else:
#             frag_ids.append(ri)
#             ri += 1
#     mask = mask_rle.astype('uint8')
#     cats = []
#     for cat in data['labels']:
#         cats.append(cat + 1)

#     seg = masks_to_seg(mask, cats, frag_ids)
#     seg = np.array(seg)
#     if output_path != None:
#         Image.fromarray(seg).save(output_path)
#     return seg , (mask , cats , frag_ids)

def load_mask_from_folder(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.tif"))
    # Convert it to a Numpy array
    return load_masks(input_files[0])

def merge_mask_by_anatomy(result):
    mask = result[0]
    anatomy = result[1]
    mask_anatomy = np.zeros([3, mask.shape[1], mask.shape[2]])
    for a in range(3):
        for i in range(len(anatomy)):
            if anatomy[i] == a + 1:
                mask_anatomy[a] += mask[i]
    mask_anatomy[mask_anatomy > 0] = 1
    return mask_anatomy

def calculate_2d_iou(vol1, vol2):
    # Calculate intersection and union
    intersection = np.logical_and(vol1, vol2).sum()
    union = np.logical_or(vol1, vol2).sum()

    # Calculate IoU
    if union == 0:
        return 0  # Avoid division by zero
    return intersection / union

def calculate_2d_hd95_from_points(vol1_points, vol2_points):
    if not vol1_points.size or not vol2_points.size:
        return np.inf

    distances = cdist(vol1_points, vol2_points, metric='euclidean').astype(np.float32)
    d1 = np.percentile(np.min(distances, axis=1), 95)
    d2 = np.percentile(np.min(distances, axis=0), 95)
    return max(d1, d2)


def calculate_2d_assd_from_points(vol1_points, vol2_points):
    if not vol1_points.size or not vol2_points.size:
        return np.inf

    distances = cdist(vol1_points, vol2_points, metric='euclidean').astype(np.float32)
    assd1 = np.mean(np.min(distances, axis=1))
    assd2 = np.mean(np.min(distances, axis=0))
    return (assd1 + assd2) / 2


def match_labels(gt_stacked_masks, pred_stacked_masks):
    # Initialize IoU dictionary
    matches = {}

    # Loop through each label in ground truth
    for label in range(len(gt_stacked_masks)):
        gt_mask = gt_stacked_masks[label]
        if gt_mask.any():
            # Calculate IoU with each label in prediction
            iou_scores = {pred_label: calculate_2d_iou(gt_mask, pred_stacked_masks[pred_label])
                          for pred_label in range(len(pred_stacked_masks)) if (pred_stacked_masks[pred_label]).any()}

            # Find the prediction label with the highest IoU
            if iou_scores:
                best_match = max(iou_scores, key=iou_scores.get)
                matches[label] = (best_match, iou_scores[best_match])

    return matches


def extract_surface_points_2D(mask, sample_size=10000):
    # Use morphological operations to find the surface (contour) of the mask
    struct = generate_binary_structure(2, 1)  # 2D connectivity
    eroded = binary_erosion(mask, structure=struct)
    surface_mask = binary_dilation(mask, structure=struct) & ~eroded

    # Extract coordinates of the surface points
    surface_points = np.argwhere(surface_mask)

    # Downsample if there are too many points
    if surface_points.shape[0] > sample_size:
        surface_points = resample(surface_points, n_samples=sample_size, random_state=2024)

    return surface_points


def calculate_sphere_radius_2D(mask):
    points = np.argwhere(mask)
    if points.size == 0:
        return np.inf  # Return inf if no points exist
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius = np.max(radii)
    return radius


def evaluate_fracture_segmentation_2D(matches, gt_mask, pred_mask):
    results = {}
    for label in matches:
        if matches[label][1] > 0:
            pred_label, _ = matches[label]
            gt_points = extract_surface_points_2D(gt_mask[label])
            pred_points = extract_surface_points_2D(pred_mask[pred_label])
            hd95 = calculate_2d_hd95_from_points(gt_points, pred_points)
            assd = calculate_2d_assd_from_points(gt_points, pred_points)
        else:
            # print("Label", label, "using maximum value.")
            radius = calculate_sphere_radius_2D(gt_mask[label])
            hd95 = 2 * radius
            assd = radius

        results[label] = (matches[label][1], hd95, assd)
    return results


def evaluate_anatomical_segmentation_2D(gt_mask, pred_mask):
    results = {}
    bone_name = ["SA", "LI", "RI"]
    for label in range(3):
        if gt_mask[label].any():
            iou = calculate_2d_iou(gt_mask[label], pred_mask[label])
            gt_points = extract_surface_points_2D(gt_mask[label])
            pred_points = extract_surface_points_2D(pred_mask[label])
            if pred_points.size:
                hd95 = calculate_2d_hd95_from_points(gt_points, pred_points)
                assd = calculate_2d_assd_from_points(gt_points, pred_points)
            else:
                radius = calculate_sphere_radius_2D(gt_mask[label])
                hd95 = 2 * radius
                assd = radius
            results[bone_name[label]] = (iou, hd95, assd)
    return results


def evaluate_2d_single_case(gt_result, pred_result, verbose=False):
    if verbose:
        print("Size =", gt_result[0].shape)
        print("Anatomy =", gt_result[1])

    # Extract and match sacrum fragments
    matches = match_labels(gt_result[0], pred_result[0])
    if verbose:
        print("Matches and IoU scores:", matches)

    # Evaluate fracture segmentation results
    if verbose:
        print("Evaluate fracture segmentation results")
    # Initialize sums and counter
    fracture_iou, fracture_hd95, fracture_assd = 0, 0, 0
    count = 0
    # Loop through results to process metrics and calculate totals
    fracture_results = evaluate_fracture_segmentation_2D(matches, gt_result[0], pred_result[0])
    for label, (iou, hd95, assd) in fracture_results.items():
        if verbose:
            print(f"Label {label}: IoU = {iou}, HD95 = {hd95}, ASSD = {assd}")
        fracture_iou += iou
        fracture_hd95 += hd95
        fracture_assd += assd
        count += 1

    # Calculate averages if there are any entries
    fracture_iou = fracture_iou / count if count != 0 else 0
    fracture_hd95 = fracture_hd95 / count if count != 0 else 1000
    fracture_assd = fracture_assd / count if count != 0 else 1000
    if verbose:
        print(f"Fracture Average IoU = {fracture_iou:.2f}, "
              f"Average HD95 = {fracture_hd95:.2f}, "
              f"Average ASSD = {fracture_assd:.2f}")

    # Evaluate anatomical segmentation results
    if verbose:
        print("Evaluate anatomical segmentation results")
    # Initialize sums and counter
    anatomical_iou, anatomical_hd95, anatomical_assd = 0, 0, 0
    count = 0
    # Loop through results to process metrics and calculate totals
    anatomical_results = evaluate_anatomical_segmentation_2D(merge_mask_by_anatomy(gt_result),
                                                             merge_mask_by_anatomy(pred_result))
    for label, (iou, hd95, assd) in anatomical_results.items():
        if verbose:
            print(f"Label {label}: IoU = {iou}, HD95 = {hd95}, ASSD = {assd}")
        anatomical_iou += iou
        anatomical_hd95 += hd95
        anatomical_assd += assd
        count += 1

    # Calculate averages if there are any entries
    anatomical_iou = anatomical_iou / count if count != 0 else 0
    anatomical_hd95 = anatomical_hd95 / count if count != 0 else 1000
    anatomical_assd = anatomical_assd / count if count != 0 else 1000
    if verbose:
        print(f"Anatomical Average IoU = {anatomical_iou:.2f}, "
              f"Average HD95 = {anatomical_hd95:.2f}, "
              f"Average ASSD = {anatomical_assd:.2f}")

    metrics_single_case = {"fracture_iou": fracture_iou,
                           "fracture_hd95": fracture_hd95,
                           "fracture_assd": fracture_assd,
                           "anatomical_iou": anatomical_iou,
                           "anatomical_hd95": anatomical_hd95,
                           "anatomical_assd": anatomical_assd}
    return metrics_single_case


if __name__ == "__main__":
    pred_mask = load_mask_from_folder(location=Path("/home/y.nawar/mmdetection/"))
    gt_mask = load_masks(Path("/scratch/dr/y.nawar/pengwin/train/output/images/x-ray/081_0000.tif"))
    metrics_single_case = evaluate_2d_single_case(gt_mask, pred_mask, verbose=True)
    print(metrics_single_case)
