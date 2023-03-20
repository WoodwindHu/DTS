import logging
import os
import pickle
import random
import shutil
import subprocess
import copy

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    if isinstance(x, np.float64) or isinstance(x, np.float32):
        return torch.tensor([x]).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('[%(asctime)s  %(filename)s %(lineno)d '
                                  '%(levelname)5s]  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res)) 
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def add_prefix_to_dict(dict, prefix):
    for key in list(dict.keys()):
        dict[prefix + key] = dict.pop(key)
    return dict


class DataReader(object):
    def __init__(self, dataloader, sampler):
        self.dataloader = dataloader
        self.sampler = sampler

    def construct_iter(self):
        self.dataloader_iter = iter(self.dataloader)

    def set_cur_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def read_data(self):
        try:
            return self.dataloader_iter.next()
        except:
            if self.sampler is not None:
                self.sampler.set_epoch(self.cur_epoch)
            self.construct_iter()
            return self.dataloader_iter.next()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_bn_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()


class NAverageMeter(object):
    """
    Contain N AverageMeter and update respectively or simultaneously
    """
    def __init__(self, n):
        self.n = n
        self.meters = [AverageMeter() for i in range(n)]

    def update(self, val, index=None, attribute='avg'):
        if isinstance(val, list) and index is None:
            assert len(val) == self.n
            for i in range(self.n):
                self.meters[i].update(val[i])
        elif isinstance(val, NAverageMeter) and index is None:
            assert val.n == self.n
            for i in range(self.n):
                self.meters[i].update(getattr(val.meters[i], attribute))
        elif not isinstance(val, list) and index is not None:
            self.meters[index].update(val)
        else:
            raise ValueError

    def aggregate_result(self):
        result = "("
        for i in range(self.n):
            result += "{:.3f},".format(self.meters[i].avg)
        result += ')'
        return result


def calculate_gradient_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def mask_dict(result_dict, mask):
    new_dict = copy.deepcopy(result_dict)
    for key, value in new_dict.items():
        new_dict[key] = value[mask]
    return new_dict


def concatenate_array_inside_dict(merged_dict, result_dict):
    for key, val in result_dict.items():
        if key not in merged_dict:
            merged_dict[key] = copy.deepcopy(val)
        else:
            merged_dict[key] = np.concatenate([merged_dict[key], copy.deepcopy(val)])

    return merged_dict

def reverse_augmentation(rois, batch_dict, is_points=False):
    '''
    modified from https://github.com/Jasonkks/mlcnet
    Args:
        rois: (B, roi_num, 7)
        batch_dict: 
            'gt_boxes': (B, M, 7)
    '''
    flip_flag_x, flip_flag_y, rotate_flag, scale_flag, object_rotate_flag, object_scale_flag \
        = False, False, False, False, False, False
    if 'world_flip_along_x' in batch_dict:
        flip_flag_x = True
        world_flip_along_x = batch_dict['world_flip_along_x'] 
    if 'world_flip_along_y' in batch_dict:
        flip_flag_y = True
        world_flip_along_y = batch_dict['world_flip_along_y'] 
    if 'world_rotation' in batch_dict:
        rotate_flag = True
        world_rotation = batch_dict['world_rotation']
    if 'world_scaling' in batch_dict:
        scale_flag = True
        world_scaling = batch_dict['world_scaling']
    if 'world_shifts' in batch_dict:
        shift_flag = True
        raise NotImplementedError
    if 'world_scaling_xyz' in batch_dict:
        raise NotImplementedError
    if 'object_rotate_noise' in batch_dict:
        object_rotate_flag = True
        object_rotate_noise = batch_dict['object_rotate_noise']
    if 'object_scale_noise' in batch_dict:
        object_scale_flag = True
        object_scale_noise = batch_dict['object_scale_noise']

    batch_size = rois.shape[0]
    gt_assignment_list = torch.zeros((batch_size, rois.shape[1]),dtype=torch.int32).cuda()

    if is_points:
        if scale_flag:
            scale_factor = 1.0 / world_scaling
            scale_factor = scale_factor.view(scale_factor.shape[0], 1, 1)
            rois *= scale_factor
    else:
        assert len(rois.shape) == 3
        gt_boxes = batch_dict['gt_boxes']
        for index in range(batch_size):
            rois[index, :, 6] = limit_period(
                rois[index, :, 6], offset=-0.5, period=2 * np.pi
                )
            # scale
            if scale_flag:
                scale_factor = 1.0 / world_scaling[index]
                rois[index,0:6] *= scale_factor
                # print('reverse scale!!!!')
            # rotation
            if rotate_flag:
                rotation_angle = - world_rotation[index].item()
                cur_rois = rois[index].cpu()
                rois[index, :, 0:3] = rotate_points_along_z(cur_rois[np.newaxis, :, 0:3], np.array([rotation_angle]))[0].cuda()
                rois[index, :, 6] += rotation_angle
            # flip
            if flip_flag_y and world_flip_along_y[index] == 1:
                rois[index,:, 0] = -rois[index,:, 0]
                rois[index,:, 6] = -(rois[index,:, 6] + np.pi)
            if flip_flag_x and world_flip_along_x[index] == 1:
                rois[index,:, 1] = -rois[index,:, 1]
                rois[index,:, 6] = rois[index,:, 6]
            if object_rotate_flag or object_scale_flag:
                cur_roi, cur_gt = rois[index], gt_boxes[index]
                k = cur_gt.__len__() - 1
                while k > 0 and cur_gt[k].sum() == 0:
                    k -= 1
                cur_gt = cur_gt[:k + 1]
                cur_gt = cur_gt.new_zeros((1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi, cur_gt[:, 0:7])  # (roi_num, k)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
                if object_rotate_flag:
                    for i in range(cur_roi.shape[0]):
                        cur_roi[i, 6] -= object_rotate_noise[index][gt_assignment[i]]
                if object_scale_flag:
                    for i in range(cur_roi.shape[0]):
                        cur_roi[i, 3:6] /= object_scale_noise[index][gt_assignment[i]]
                rois[index] = cur_roi
                gt_assignment_list[index] = gt_assignment

    return rois, gt_assignment_list