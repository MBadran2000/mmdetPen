from albumentations import Lambda
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS , LOOPS , METRICS
# from mmpretrain.registry import  ME
from peng_utils import *
import torch
from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.amp import autocast
from mmengine.evaluator import Evaluator
from mmengine.logging import HistoryBuffer, print_log ,MMLogger
from mmengine.utils import is_list_of
from pengwin_eval import evaluate_2d_single_case , json_to_tif
# import numpy as np
from typing import Dict, List, Sequence, Union
from torch.utils.data import DataLoader
from mmengine.structures import BaseDataElement
# from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric


        
@METRICS.register_module()
class PengwinMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process(self, data_batch, data_samples):
        # flag = True
        for output in data_samples:
            gt = self.decode_pred_instances(pred_instances = output['gt_instances'])
            gt['scores'] = np.ones_like(gt['labels'])
            mask_gt = self.json_to_tif(gt)
            preds = self.decode_pred_instances(output['pred_instances'], pred=True)
            mask_preds = self.json_to_tif(preds)
            # metrics = evaluate_2d_single_case(mask_gt, mask_preds, verbose=False)
            metrics = evaluate_2d_single_case(mask_gt , mask_preds)
            self.results.append(metrics)

    def decode_pred_instances(self,pred_instances , pred = False):
        # Access pred_instances
        # pred_instances = sample.pred_instances
        # Initialize an empty dictionary to store the arrays
        decoded_pred = {}

        # Extract bounding boxes, labels, and scores
        decoded_pred['bboxes'] = pred_instances['bboxes'].cpu().numpy()
        decoded_pred['labels'] = pred_instances['labels'].cpu().numpy()
        if pred:
            decoded_pred['scores'] = pred_instances['scores'].cpu().numpy()
        if pred:
            decoded_pred['masks'] = np.array(pred_instances['masks'].cpu()).astype('uint8')
        else:
            decoded_pred['masks'] = np.array(pred_instances['masks'])
        return decoded_pred

    def json_to_tif(self , data , threshold = 0.5):
        if threshold > max(data['scores']):
            threshold = min(data['scores'])

        filtered_indices = [i for i, score in enumerate(data['scores']) if score >= threshold]

        data = {
            "labels": [data['labels'][i] for i in filtered_indices],
            "scores": [data['scores'][i] for i in filtered_indices],
            "bboxes": [data['bboxes'][i] for i in filtered_indices],
            "masks": [data['masks'][i] for i in filtered_indices]
        }
        mask_rle = np.array(data['masks'])
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

        mask = mask_rle.astype('uint8')
        cats = []
        for cat in data['labels']:
            cats.append(cat + 1)

        # seg = masks_to_seg(mask, cats, frag_ids)

        return (mask , cats , frag_ids)
    
    def compute_metrics(self, results):
        print('Computing Metrics')
        logger: MMLogger = MMLogger.get_current_instance()
        # gts, preds = zip(*results)

        count = len(results)
        fracture_iou , fracture_assd , fracture_hd95 = 0 , 0, 0
        anatomical_iou , anatomical_assd , anatomical_hd95 = 0 , 0, 0
        for metrics in results:
            fracture_iou += metrics['fracture_iou']
            fracture_assd += metrics['fracture_assd']
            fracture_hd95 += metrics['fracture_hd95']
            anatomical_iou += metrics['anatomical_iou']
            anatomical_assd += metrics['anatomical_assd']
            anatomical_hd95 += metrics['anatomical_hd95']
        
        metrics = {'fracture_iou' : fracture_iou / count,
                   'fracture_assd' : fracture_assd / count,
                   'fracture_hd95' : fracture_hd95 / count,
                   'anatomical_iou' : anatomical_iou / count,
                   'anatomical_assd' : anatomical_assd / count,
                   'anatomical_hd95' : anatomical_hd95 / count}

        logger.info(f'fracture_iou : {fracture_iou / count :.3f} '
                    f'fracture_assd : {fracture_assd / count:.3f} '
                    f'fracture_hd95 : {fracture_hd95 / count:.3f} ' 
                    f'anatomical_iou : {anatomical_iou / count:.3f} '
                    f'anatomical_assd : {anatomical_assd / count:.3f} '
                    f'anatomical_hd95 : {anatomical_hd95 / count:.3f} ')
        return metrics
 
@TRANSFORMS.register_module()
class CustomGaussianContrast:
    def __init__(self, alpha=(0.6, 1.4), sigma=(0.1, 0.5), max_value=1):
        self.alpha = alpha
        self.sigma = sigma
        self.max_value = max_value

    def __call__(self, results):
        results['img'] = gaussian_contrast_fn(
            results['img'][None],  # Add batch dimension
            alpha=self.alpha,
            sigma=self.sigma,
            max_value=self.max_value
        )[0]  # Remove batch dimension
        return results

@TRANSFORMS.register_module()
class CustomNegLogTransform:
    def __init__(self, epsilon=0.001):
        self.epsilon = epsilon

    def __call__(self, results):
        # print(results.keys())
        results['img'] = neglog_fn(results['img'], epsilon=self.epsilon)
        return results
    
@TRANSFORMS.register_module()
class CustomWindowTransform:
    def __init__(self, lower=0.01, upper=0.99, convert=True):
        self.lower = lower
        self.upper = upper
        self.convert = convert

    def __call__(self, results):
        results['img'] = window_(
            results['img'],
            lower=self.lower,
            upper=self.upper,
            convert=self.convert
        )
        return results
        
@LOOPS.register_module()
class MyValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16
        self.val_loss: Dict[str, HistoryBuffer] = dict()
        self.metrics_pengwin = {}
        # self.fracture_iou= []
        # self.fracture_hd95 = []
        # self.fracture_assd = []
        # self.anatomical_iou, self.anatomical_hd95, self.anatomical_assd = [], [],[]

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        # clear val loss
        self.val_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metric
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)

        # self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.metrics_pengwin.update(metrics)
        self.runner.call_hook('after_val_epoch', metrics=self.metrics_pengwin)

        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.
        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)

        for output  in outputs:
            gt = decode_pred_instances(output.gt_instances, pred= False)
            gt['scores']= np.ones_like(gt['labels'])
            seg_gt , (mask_gt) = json_to_tif(gt)
            preds = decode_pred_instances(output.pred_instances , pred = True)
            seg_preds , (mask_preds) = json_to_tif(preds)
            # print(mask_gt[0].shape , mask_preds[0].shape)
            metrics = evaluate_2d_single_case(mask_gt , mask_preds , verbose=False)
            self.update_pengwin(metrics)

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        
    def update_pengwin(self , metrics):
        for key in metrics.keys():
            if key in self.metrics_pengwin.keys():
                self.metrics_pengwin[key] += metrics[key]
            else:
                self.metrics_pengwin[key] = metrics[key]
    
    def finalize(self):
        for key in self.metrics_pengwin.keys():
            self.metrics_pengwin[key] = self.metrics_pengwin[key] / 10000
        
def decode_pred_instances(pred_instances , pred = False):
    # Access pred_instances
    # pred_instances = sample.pred_instances
    
    # Initialize an empty dictionary to store the arrays
    decoded_pred = {}

    # Extract bounding boxes, labels, and scores
    if hasattr(pred_instances, 'bboxes'):
        decoded_pred['bboxes'] = pred_instances.bboxes.cpu().numpy()
    if hasattr(pred_instances, 'labels'):
        decoded_pred['labels'] = pred_instances.labels.cpu().numpy()
    if hasattr(pred_instances, 'scores'):
        decoded_pred['scores'] = pred_instances.scores.cpu().numpy()

    # If there are masks (e.g., in instance segmentation)
    if hasattr(pred_instances, 'masks'):
        if pred:
            decoded_pred['masks'] = np.array(pred_instances.masks.cpu()).astype('uint8')
        else:
            decoded_pred['masks'] = np.array(pred_instances.masks)

    return decoded_pred

def _update_losses(outputs: list, losses: dict) -> Tuple[list, dict]:
    """Update and record the losses of the network.

    Args:
        outputs (list): The outputs of the network.
        losses (dict): The losses of the network.

    Returns:
        list: The updated outputs of the network.
        dict: The updated losses of the network.
    """
    if isinstance(outputs[-1],
                  BaseDataElement) and outputs[-1].keys() == ['loss']:
        loss = outputs[-1].loss  # type: ignore
        outputs = outputs[:-1]
    else:
        loss = dict()

    for loss_name, loss_value in loss.items():
        if loss_name not in losses:
            losses[loss_name] = HistoryBuffer()
        if isinstance(loss_value, torch.Tensor):
            losses[loss_name].update(loss_value.item())
        elif is_list_of(loss_value, torch.Tensor):
            for loss_value_i in loss_value:
                losses[loss_name].update(loss_value_i.item())
    return outputs, losses

def _parse_losses(losses: Dict[str, HistoryBuffer],
                  stage: str) -> Dict[str, float]:
    """Parses the raw losses of the network.

    Args:
        losses (dict): raw losses of the network.
        stage (str): The stage of loss, e.g., 'val' or 'test'.

    Returns:
        dict[str, float]: The key is the loss name, and the value is the
        average loss.
    """
    all_loss = 0
    loss_dict: Dict[str, float] = dict()

    for loss_name, loss_value in losses.items():
        avg_loss = loss_value.mean()
        loss_dict[loss_name] = avg_loss
        if 'loss' in loss_name:
            all_loss += avg_loss

    loss_dict[f'{stage}_loss'] = all_loss
    return loss_dict
