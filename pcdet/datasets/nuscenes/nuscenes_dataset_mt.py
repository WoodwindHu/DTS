import copy
import imp
import numpy as np
from ...utils import common_utils
from .nuscenes_dataset import NuScenesDataset
from collections import defaultdict
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from pathlib import Path



class NuScenesDatasetMT(NuScenesDataset):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)

        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']).stem,
            'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

            if self.dataset_cfg.get('SHIFT_COOR', None):
                input_dict['gt_boxes'][:, 0:3] += self.dataset_cfg.SHIFT_COOR

            if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
                input_dict['gt_boxes'] = None

            # for debug only
            # gt_boxes_mask = np.array([n in self.class_names for n in input_dict['gt_names']], dtype=np.bool_)
            # debug_dict = {'gt_boxes': copy.deepcopy(input_dict['gt_boxes'][gt_boxes_mask])}

        if self.dataset_cfg.get('FOV_POINTS_ONLY', None):
            input_dict['points'] = self.extract_fov_data(
                input_dict['points'], self.dataset_cfg.FOV_DEGREE, self.dataset_cfg.FOV_ANGLE
            )
            if input_dict['gt_boxes'] is not None:
                fov_gt_flag = self.extract_fov_gt(
                    input_dict['gt_boxes'], self.dataset_cfg.FOV_DEGREE, self.dataset_cfg.FOV_ANGLE
                )
                input_dict.update({
                    'gt_names': input_dict['gt_names'][fov_gt_flag],
                    'gt_boxes': input_dict['gt_boxes'][fov_gt_flag],
                })

        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            gt_boxes = input_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            input_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in input_dict and not self.dataset_cfg.get('USE_PSEUDO_LABEL', None):
            input_dict['gt_boxes'] = input_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6]]

        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        data_dict1 = copy.deepcopy(input_dict)
        data_dict2 = copy.deepcopy(input_dict)
        data_dict1 = self.prepare_data(data_dict=data_dict1)
        if isinstance(data_dict1, list):    # len(data_dict['gt_boxes']) == 0
            return data_dict1
        data_dict2 = self.prepare_data_teacher(data_dict=data_dict2)
        if isinstance(data_dict2, list):    # len(data_dict['gt_boxes']) == 0
            return data_dict2
        
        output = [data_dict1, data_dict2]
        return output

    def prepare_data_teacher(self, data_dict):
        """
        Args:
            data_dict:
                points: optional, (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            # filter gt_boxes without points
            num_points_in_gt = data_dict.get('num_points_in_gt', None)
            if num_points_in_gt is None:
                num_points_in_gt = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(data_dict['points'][:, :3]),
                    torch.from_numpy(data_dict['gt_boxes'][:, :7])).numpy().sum(axis=1)

            mask = (num_points_in_gt >= self.dataset_cfg.get('MIN_POINTS_OF_GT', 1))
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
            data_dict['gt_names'] = data_dict['gt_names'][mask]
            if 'gt_classes' in data_dict:
                data_dict['gt_classes'] = data_dict['gt_classes'][mask]
                data_dict['gt_scores'] = data_dict['gt_scores'][mask]

            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            ## note: teacher has no augmentation
            if self.dataset_cfg.get('TEACHER_RBRS', None):
                augmentor_configs = self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST
                aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
                    else augmentor_configs.AUG_CONFIG_LIST
                for cur_cfg in aug_config_list:
                    if cur_cfg.NAME == 'random_beam_downsample':
                        beam = cur_cfg.BEAM
                        beam_prob = cur_cfg.BEAM_PROB
                points = data_dict['points']
                beam_label = data_dict['beam_labels']
                beam_mask = np.random.rand(beam) < beam_prob
                points_mask = beam_mask[beam_label]
                data_dict['points'] = points[points_mask]


            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]

            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]
            if len(data_dict['gt_boxes']) == 0:
                new_index = np.random.randint(self.__len__())
                return self.__getitem__(new_index)

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            # for pseudo label has ignore labels.
            if 'gt_classes' not in data_dict:
                gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            else:
                gt_classes = data_dict['gt_classes'][selected]
                data_dict['gt_scores'] = data_dict['gt_scores'][selected]
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

            if data_dict.get('gt_boxes2d', None) is not None:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][selected]

        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        data_dict.pop('gt_classes', None)

        return data_dict


    @staticmethod
    def collate_batch(batch_list, _unused=False):
        def collate_fn(batch_list):
            data_dict = defaultdict(list)
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)
            ret = {}

            for key, val in data_dict.items():
                try:
                    if key in ['voxels', 'voxel_num_points']:
                        ret[key] = np.concatenate(val, axis=0)
                    elif key in ['points', 'voxel_coords']:
                        coors = []
                        for i, coor in enumerate(val):
                            coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                            coors.append(coor_pad)
                        ret[key] = np.concatenate(coors, axis=0)
                    elif key in ['gt_boxes']:
                        max_gt = max([len(x) for x in val])
                        batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                        for k in range(batch_size):
                            batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_gt_boxes3d
                    elif key in ['gt_scores']:
                        max_gt = max([len(x) for x in val])
                        batch_scores = np.zeros((batch_size, max_gt), dtype=np.float32)
                        for k in range(batch_size):
                            batch_scores[k, :val[k].__len__()] = val[k]
                        ret[key] = batch_scores
                    elif key in ['gt_boxes2d']:
                        max_boxes = 0
                        max_boxes = max([len(x) for x in val])
                        batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                        for k in range(batch_size):
                            if val[k].size > 0:
                                batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                        ret[key] = batch_boxes2d
                    elif key in ["images", "depth_maps"]:
                        # Get largest image size (H, W)
                        max_h = 0
                        max_w = 0
                        for image in val:
                            max_h = max(max_h, image.shape[0])
                            max_w = max(max_w, image.shape[1])

                        # Change size of images
                        images = []
                        for image in val:
                            pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                            pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                            pad_width = (pad_h, pad_w)
                            # Pad with nan, to be replaced later in the pipeline.
                            pad_value = np.nan

                            if key == "images":
                                pad_width = (pad_h, pad_w, (0, 0))
                            elif key == "depth_maps":
                                pad_width = (pad_h, pad_w)

                            image_pad = np.pad(image,
                                            pad_width=pad_width,
                                            mode='constant',
                                            constant_values=pad_value)

                            images.append(image_pad)
                        ret[key] = np.stack(images, axis=0)
                    elif key in ['object_scale_noise', 'object_rotate_noise']:
                        max_noise = max([len(x) for x in val])
                        batch_noise = np.zeros((batch_size, max_noise), dtype=np.float32)
                        for k in range(batch_size):
                            batch_noise[k, :val[k].__len__()] = val[k]
                        ret[key] = batch_noise
                    elif key in ['beam_labels']:
                        continue
                    else:
                        ret[key] = np.stack(val, axis=0)
                except:
                    print('Error in collate_batch: key=%s' % key)
                    raise TypeError

            ret['batch_size'] = batch_size
            return ret

        if isinstance(batch_list[0], dict):
            return collate_fn(batch_list)

        else:
            assert isinstance(batch_list[0], list)
            batch_list1 = [x[0] for x in batch_list]
            batch_list2 = [x[1] for x in batch_list]
            ret1 = collate_fn(batch_list1)
            ret2 = collate_fn(batch_list2)
            return [ret1, ret2]
    

