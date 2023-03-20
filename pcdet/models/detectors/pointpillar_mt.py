from .detector3d_template import Detector3DTemplate
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..model_utils.model_nms_utils import class_agnostic_nms
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from torch.nn.modules.batchnorm import _BatchNorm
from ...utils import loss_utils, common_utils
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import numpy as np


class PointPillarMT(Detector3DTemplate):
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
        assert self.training
        
        batch_dict, batch_target2 = batch_dict
        batch_dict['dataset_cfg'] = self.dataset.dataset_cfg
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        loss1, tb_dict1, disp_dict1 = self.get_training_loss()

        if self.model_cfg.CONSISTENCY_LOSS.get('inter_graph_loss_weight', 0) > 0 \
            or self.model_cfg.CONSISTENCY_LOSS.get('intra_graph_loss_weight', 0) > 0 \
            or self.model_cfg.CONSISTENCY_LOSS.get('contrastive_loss_weight', 0) > 0 \
            or self.model_cfg.CONSISTENCY_LOSS.get('object_loss_weight', 0) > 0:
            loss2, tb_dict2 = self.get_graph_loss(batch_dict, batch_target2, self.model_cfg.get('GRAPH_MATRIX_REWEIGHT', True))
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
        return ret_dict, tb_dict1, disp_dict1, batch_dict

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
