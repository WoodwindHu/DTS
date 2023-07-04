from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')



def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    if isinstance(batch_dict, dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            elif key in ['frame_id', 'metadata', 'calib']:
                continue
            elif key in ['images']:
                batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
            elif key in ['image_shape']:
                batch_dict[key] = torch.from_numpy(val).int().cuda()
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()
    else:
        assert isinstance(batch_dict, list)
        for batch in batch_dict:
            for key, val in batch.items():
                if not isinstance(val, np.ndarray):
                    continue
                elif key in ['frame_id', 'metadata', 'calib']:
                    continue
                elif key in ['images']:
                    batch[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
                elif key in ['image_shape']:
                    batch[key] = torch.from_numpy(val).int().cuda()
                else:
                    batch[key] = torch.from_numpy(val).float().cuda()
        


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

def model_fn_decorator_for_mt():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict', 'batch_target1'])

    def model_func(model, batch_dict, ema_model=None, cur_epoch=None):
        load_data_to_gpu(batch_dict)
        batch_target1, batch_target2 = batch_dict

        # add tag for target domain
        batch_target1['batch_type'] = 'target'
        batch_target2['batch_type'] = 'target'
        
        # forward teacher model first
        batch_target2 = ema_model(batch_target2, return_batch_dict=True)
        batch_target_teacher = {}
        for key, val in batch_target2.items():
            if key in ['batch_size','batch_cls_preds','multihead_label_mapping',
                            'batch_box_preds','cls_preds_normalized',
                            'batch_index','has_class_labels', 'rois', 'roi_labels',
                            'roi_scores','roi_head_features','batch_pred_labels', 
                            'world_scaling', 'world_rotation', 'world_flip_along_y', 
                            'world_flip_along_x', 'gt_boxes', 'rois_mt', 'roi_labels_mt',
                            'roi_scores_mt',]:
                batch_target_teacher[key] = val 
        
        # forward student model
        # ret_dict, tb_dict, disp_dict = model([batch_target1, batch_target2], is_ema=False, cur_epoch=cur_epoch)
        ret_dict, tb_dict, disp_dict, batch_target1 = model([batch_target1, batch_target_teacher], is_ema=False, cur_epoch=cur_epoch)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict, batch_target1)

    return model_func