from functools import partial
import numpy as np
from . import augmentor_utils, database_sampler
from ...utils import common_utils, downsample_utils


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger
        self.augmentor_configs = augmentor_configs

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
    
    def label_point_cloud_beam(self, polar_image, points, beam=32):
        if polar_image.shape[0] <= beam:
            print("too small point cloud!")
            return np.arange(polar_image.shape[0])
        beam_label, centroids = downsample_utils.beam_label(polar_image[:,1], beam)
        idx = np.argsort(centroids)
        rev_idx = np.zeros_like(idx)
        for i, t in enumerate(idx):
            rev_idx[t] = i
        beam_label = rev_idx[beam_label]
        return beam_label

    def get_polar_image(self, points):
        theta, phi = downsample_utils.compute_angles(points[:,:3])
        r = np.sqrt(np.sum(points[:,:3]**2, axis=1))
        polar_image = points.copy()
        polar_image[:,0] = phi 
        polar_image[:,1] = theta
        polar_image[:,2] = r 
        return polar_image

    def random_beam_upsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_upsample, config=config)
        points = data_dict['points']
        polar_image = self.get_polar_image(points)
        beam_label = self.label_point_cloud_beam(polar_image, points, config['BEAM'])
        new_pcs = [points]
        phi = polar_image[:,0]
        for i in range(config['BEAM'] - 1):
            if np.random.rand() < config['BEAM_PROB'][i]:
                cur_beam_mask = (beam_label == i)
                next_beam_mask = (beam_label == i + 1)
                delta_phi = np.abs(phi[cur_beam_mask, np.newaxis] - phi[np.newaxis, next_beam_mask])
                corr_idx = np.argmin(delta_phi,1)
                min_delta = np.min(delta_phi,1)
                mask = min_delta < config['PHI_THRESHOLD']
                cur_beam = polar_image[cur_beam_mask][mask]
                next_beam = polar_image[next_beam_mask][corr_idx[mask]]
                new_beam = (cur_beam + next_beam)/2
                new_pc = new_beam.copy()
                new_pc[:,0] = np.cos(new_beam[:,1]) * np.cos(new_beam[:,0]) * new_beam[:,2]
                new_pc[:,1] = np.cos(new_beam[:,1]) * np.sin(new_beam[:,0]) * new_beam[:,2]
                new_pc[:,2] = np.sin(new_beam[:,1]) * new_beam[:,2]
                new_pcs.append(new_pc)
        data_dict['points'] = np.concatenate(new_pcs,0)
        return data_dict

    def random_beam_downsample(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_beam_downsample, config=config)
        points = data_dict['points']
        if 'beam_labels' in data_dict: # for waymo and kitti datasets
            beam_label = data_dict['beam_labels']
        else:
            polar_image = self.get_polar_image(points)
            beam_label = self.label_point_cloud_beam(polar_image, points, config['BEAM'])
        beam_mask = np.random.rand(config['BEAM']) < config['BEAM_PROB']
        points_mask = beam_mask[beam_label]
        data_dict['points'] = points[points_mask]

        return data_dict

    def random_object_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_rotation, config=config)

        gt_boxes, points, object_rotate_noise = augmentor_utils.rotate_objects(
            data_dict['gt_boxes'],
            data_dict['points'],
            data_dict['gt_boxes_mask'],
            rotation_perturb=config['ROT_UNIFORM_NOISE'],
            prob=config['ROT_PROB'],
            num_try=50
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['object_rotate_noise'] = object_rotate_noise
        return data_dict

    def random_object_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_object_scaling, config=config)
        points, gt_boxes, object_scale_noise = augmentor_utils.scale_pre_object(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            scale_perturb=config['SCALE_UNIFORM_NOISE']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['object_scale_noise'] = object_scale_noise
        return data_dict

    def random_world_sampling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_sampling, config=config)
        gt_boxes, points, gt_boxes_mask = augmentor_utils.global_sampling(
            data_dict['gt_boxes'], data_dict['points'],
            gt_boxes_mask=data_dict['gt_boxes_mask'],
            sample_ratio_range=config['WORLD_SAMPLE_RATIO'],
            prob=config['PROB']
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_boxes_mask'] = gt_boxes_mask
        data_dict['points'] = points
        return data_dict

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )
            data_dict['world_flip_along_%s' % cur_axis] = enable

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rotation = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['world_rotation'] = noise_rotation
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['world_scaling'] = noise_scale
        return data_dict

    def normalize_object_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.normalize_object_size, config=config)
        points, gt_boxes = augmentor_utils.normalize_object_size(
            data_dict['gt_boxes'], data_dict['points'], data_dict['gt_boxes_mask'], config['SIZE_RES']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict

    def re_prepare(self, augmentor_configs=None, intensity=None):
        self.data_augmentor_queue = []

        if augmentor_configs is None:
            augmentor_configs = self.augmentor_configs

        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            # scale data augmentation intensity
            if intensity is not None:
                cur_cfg = self.adjust_augment_intensity(cur_cfg, intensity)
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def adjust_augment_intensity(self, config, intensity):
        adjust_map = {
            'random_object_scaling': 'SCALE_UNIFORM_NOISE',
            'random_object_rotation': 'ROT_UNIFORM_NOISE',
            'random_world_rotation': 'WORLD_ROT_ANGLE',
            'random_world_scaling': 'WORLD_SCALE_RANGE',
        }

        def cal_new_intensity(config, flag):
            origin_intensity_list = config.get(adjust_map[config.NAME])
            assert len(origin_intensity_list) == 2
            assert np.isclose(flag - origin_intensity_list[0], origin_intensity_list[1] - flag)
            
            noise = origin_intensity_list[1] - flag
            new_noise = noise * intensity
            new_intensity_list = [flag - new_noise, new_noise + flag]
            return new_intensity_list

        if config.NAME not in adjust_map:
            return config
        
        # for data augmentations that init with 1
        if config.NAME in ["random_object_scaling", "random_world_scaling"]:
            new_intensity_list = cal_new_intensity(config, flag=1)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        elif config.NAME in ['random_object_rotation', 'random_world_rotation']:
            new_intensity_list = cal_new_intensity(config, flag=0)
            setattr(config, adjust_map[config.NAME], new_intensity_list)
            return config
        else:
            raise NotImplementedError
