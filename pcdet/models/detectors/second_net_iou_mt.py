import torch
import torch.nn as nn
import torch.nn.functional as F
from .detector3d_template import Detector3DTemplate
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from torch.nn.modules.batchnorm import _BatchNorm
from ...utils import loss_utils, common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import numpy as np


class SECONDNetIoUMT(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def set_momemtum_value_for_bn(self, momemtum=0.1):
        def apply_fn(m, momemtum=momemtum):
            if isinstance(m, _BatchNorm):
                m.momentum = momemtum
        self.apply(apply_fn)

    def reset_bn_stats(self):
        def apply_fn(m):
            if isinstance(m, _BatchNorm):
                m.reset_running_stats()
        self.apply(apply_fn)

    def forward(self, batch_dict, is_ema=True, cur_epoch=None, ema_model=None, return_batch_dict=False):
        if is_ema:
            if isinstance(batch_dict, dict):
                batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
                for cur_module in self.module_list:
                    batch_dict = cur_module(batch_dict)
                if return_batch_dict:
                    return batch_dict
                else:
                    pred_dicts, recall_dicts = self.post_processing(batch_dict)
                    return pred_dicts, recall_dicts
            else:
                raise NotImplementedError
        assert isinstance(batch_dict, list)
        assert len(batch_dict) == 2
        batch_merge, batch_target2 = batch_dict
        batch_merge['dataset_cfg'] = self.dataset.dataset_cfg

        if batch_merge['batch_type'] == 'merge':
            for cur_module in self.module_list:
                batch_merge = cur_module(batch_merge)
            # split source and target batch
            __, batch_target1 = self.split_batch_dicts(batch_merge)
        else:
            # print("batch not merge!!!!!!!!!")
            for cur_module in self.module_list:
                batch_merge = cur_module(batch_merge)
            batch_target1 = batch_merge

        

        assert self.training
        
        loss1, tb_dict1, disp_dict1 = self.get_training_loss()

        if self.model_cfg.CONSISTENCY_LOSS.get('inter_graph_loss_weight',0) > 0 \
            or self.model_cfg.CONSISTENCY_LOSS.get('intra_graph_loss_weight',0) > 0 \
            or self.model_cfg.CONSISTENCY_LOSS.get('contrastive_loss_weight',0) > 0 \
            or self.model_cfg.CONSISTENCY_LOSS.get('object_loss_weight', 0) > 0:
            loss2, tb_dict2 = self.get_graph_loss(batch_target1, batch_target2, self.model_cfg.get('GRAPH_MATRIX_REWEIGHT', True))
        else:
            loss2, tb_dict2 = 0, {}

        tb_dict1.update(tb_dict2)

        source_loss_weight = self.model_cfg.get('SOURCE_LOSS_WEIGHT', 1.0)
        if self.model_cfg.get('SOURCE_LOSS_SCHEDULE', 'off') == 'exp':
            multiplier = np.exp(-cur_epoch)
        else:
            multiplier = 1.0
        loss = loss1 * source_loss_weight * multiplier + loss2
        
        ret_dict = {
            'loss': loss
        }

        return ret_dict, tb_dict1, disp_dict1, batch_target1

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict

    @staticmethod
    def cal_scores_by_npoints(cls_scores, iou_scores, num_points_in_gt, cls_thresh=10, iou_thresh=100):
        """
        Args:
            cls_scores: (N)
            iou_scores: (N)
            num_points_in_gt: (N, 7+c)
            cls_thresh: scalar
            iou_thresh: scalar
        """
        assert iou_thresh >= cls_thresh
        alpha = torch.zeros(cls_scores.shape, dtype=torch.float32).cuda()
        alpha[num_points_in_gt <= cls_thresh] = 0
        alpha[num_points_in_gt >= iou_thresh] = 1
        
        mask = ((num_points_in_gt > cls_thresh) & (num_points_in_gt < iou_thresh))
        alpha[mask] = (num_points_in_gt[mask] - 10) / (iou_thresh - cls_thresh)
        
        scores = (1 - alpha) * cls_scores + alpha * iou_scores

        return scores

    def set_nms_score_by_class(self, iou_preds, cls_preds, label_preds, score_by_class):
        n_classes = torch.unique(label_preds).shape[0]
        nms_scores = torch.zeros(iou_preds.shape, dtype=torch.float32).cuda()
        for i in range(n_classes):
            mask = label_preds == (i + 1)
            class_name = self.class_names[i]
            score_type = score_by_class[class_name]
            if score_type == 'iou':
                nms_scores[mask] = iou_preds[mask]
            elif score_type == 'cls':
                nms_scores[mask] = cls_preds[mask]
            else:
                raise NotImplementedError

        return nms_scores

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            iou_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = batch_dict['roi_scores'][batch_mask]

            src_iou_preds = iou_preds
            src_box_preds = box_preds
            src_cls_preds = cls_preds
            assert iou_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                iou_preds = torch.sigmoid(iou_preds)
                cls_preds = torch.sigmoid(cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                iou_preds, label_preds = torch.max(iou_preds, dim=-1)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels', False) else label_preds + 1

                if post_process_cfg.NMS_CONFIG.get('SCORE_BY_CLASS', None) and \
                        post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'score_by_class':
                    nms_scores = self.set_nms_score_by_class(
                        iou_preds, cls_preds, label_preds, post_process_cfg.NMS_CONFIG.SCORE_BY_CLASS
                    )
                elif post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) == 'iou' or \
                        post_process_cfg.NMS_CONFIG.get('SCORE_TYPE', None) is None:
                    nms_scores = iou_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'cls':
                    nms_scores = cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'weighted_iou_cls':
                    nms_scores = post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.iou * iou_preds + \
                                 post_process_cfg.NMS_CONFIG.SCORE_WEIGHTS.cls * cls_preds
                elif post_process_cfg.NMS_CONFIG.SCORE_TYPE == 'num_pts_iou_cls':
                    point_mask = (batch_dict['points'][:, 0] == batch_mask)
                    batch_points = batch_dict['points'][point_mask][:, 1:4]

                    num_pts_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                        batch_points.cpu(), box_preds[:, 0:7].cpu()
                    ).sum(dim=1).float().cuda()
                    
                    score_thresh_cfg = post_process_cfg.NMS_CONFIG.SCORE_THRESH
                    nms_scores = self.cal_scores_by_npoints(
                        cls_preds, iou_preds, num_pts_in_gt, 
                        score_thresh_cfg.cls, score_thresh_cfg.iou
                    )
                else:
                    raise NotImplementedError

                selected, selected_scores = class_agnostic_nms(
                    box_scores=nms_scores, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    raise NotImplementedError

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels,
                'pred_cls_scores': cls_preds[selected],
                'pred_iou_scores': iou_preds[selected]
            }

            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict
