import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def cat(x, y, axis=0):
    return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def write_info(args, fp):
    data = {
        'timestamp': str(datetime.now()),
        'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
        'args': vars(args)
    }
    with open(fp, 'a') as f:
        json.dump(data, f, indent=4, separators=(',', ': '))
        f.write("\n")


def load_config(key=None):
    path = os.path.join('setup', 'config.cfg')
    with open(path) as f:
        data = json.load(f)
    if key is not None:
        return data[key]
    return data


def make_dir(dir_path):
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
    fpath = os.path.join(dir_path, f'*.{filetype}')
    fpaths = glob.glob(fpath, recursive=True)
    if sort:
        return sorted(fpaths)
    return fpaths


def prefill_memory(obses, capacity, obs_shape):
    """Reserves memory for replay buffer"""
    c, h, w = obs_shape
    for _ in range(capacity):
        frame = np.ones((3, h, w), dtype=np.uint8)
        obses.append(frame)
    return obses


class ReplayBuffer(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, prefill=True):
        self.capacity = capacity
        self.batch_size = batch_size

        self._obses = []
        if prefill:
            self._obses = prefill_memory(self._obses, capacity, obs_shape)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        obses = (obs, next_obs)
        if self.idx >= len(self._obses):
            self._obses.append(obses)
        else:
            self._obses[self.idx] = (obses)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size
        return np.random.randint(
            0, self.capacity if self.full else self.idx, size=n
        )

    def _encode_obses(self, idxs):
        obses, next_obses = [], []
        for i in idxs:
            obs, next_obs = self._obses[i]
            obses.append(np.array(obs, copy=False))
            next_obses.append(np.array(next_obs, copy=False))
        return np.array(obses), np.array(next_obses)

    def sample_soda(self, n=None):
        idxs = self._get_idxs(n)
        obs, _ = self._encode_obses(idxs)
        return torch.as_tensor(obs).cuda().float()

    def __sample__(self, n=None):
        idxs = self._get_idxs(n)

        obs, next_obs = self._encode_obses(idxs)
        obs = torch.as_tensor(obs).cuda().float()
        next_obs = torch.as_tensor(next_obs).cuda().float()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        return obs, actions, rewards, next_obs, not_dones

    def sample_curl(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        pos = augmentations.random_crop(obs.clone())
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones, pos

    def sample_drq(self, n=None, pad=4):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_shift(obs, pad)
        next_obs = augmentations.random_shift(next_obs, pad)

        return obs, actions, rewards, next_obs, not_dones

    def sample_svea(self, n=None, pad=4):
        return self.sample_drq(n=n, pad=pad)

    def sample(self, n=None):
        obs, actions, rewards, next_obs, not_dones = self.__sample__(n=n)
        obs = augmentations.random_crop(obs)
        next_obs = augmentations.random_crop(next_obs)

        return obs, actions, rewards, next_obs, not_dones


class LazyFrames(object):
    def __init__(self, frames, extremely_lazy=True):
        self._frames = frames
        self._extremely_lazy = extremely_lazy
        self._out = None

    @property
    def frames(self):
        return self._frames

    def _force(self):
        if self._extremely_lazy:
            return np.concatenate(self._frames, axis=0)
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        if self._extremely_lazy:
            return len(self._frames)
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        if self.extremely_lazy:
            return len(self._frames)
        frames = self._force()
        return frames.shape[0]//3

    def frame(self, i):
        return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
    """Returns total number of params in a network"""
    count = sum(p.numel() for p in net.parameters())
    if as_int:
        return count
    return f'{count:,}'


from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide

sam_checkpoint = './sam_vit_b_01ec64.pth'
sam_model_type = 'vit_b'
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint).cuda()
sam_predictor = SamPredictor(sam)
sam_resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def generate_boxes(image):
    height, width = image.shape[:2]

    center = (width // 2, height // 2)
    box_width = width // 2.5
    box_heigth = height // 2.5

    # fb_center = (width // 4, height // 2)
    # sb_center = (width * 3 // 4, height // 2)
    # box_width = width // 4
    # box_heigth = height // 2

    return torch.tensor([
        # [fb_center[0] - box_width, fb_center[1] - box_heigth, fb_center[0] + box_width, fb_center[1] + box_heigth],
        # [sb_center[0] - box_width, sb_center[1] - box_heigth, sb_center[0] + box_width, sb_center[1] + box_heigth],
        [center[0] - box_width, center[1] - box_heigth, center[0] + box_width, center[1] + box_heigth],
    ], device=sam.device)

def extract_batch_saliency_map(images, mode=1):
    batched_input = [{
        'image': prepare_image(image, sam_resize_transform, sam),
        'boxes': sam_resize_transform.apply_boxes_torch(generate_boxes(image), image.shape[:2]),
        'original_size': image.shape[:2]
    } for image in images]
    batched_output = sam(batched_input, multimask_output=False)
    saliency_maps = [output['masks'][0].cpu().numpy() for output in batched_output]
    saliency_maps = [np.squeeze(saliency_map, axis=0) for saliency_map in saliency_maps]
    return saliency_maps

def extract_saliency_map(image, mode=1):
    # saliency_map = np.zeros_like(image)
    if mode == 1: # use YOLO + SAM
        boxes = generate_boxes(image)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
        sam_predictor.set_image(image)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False)
        saliency_map = masks[0].cpu().numpy()
        saliency_map = np.squeeze(saliency_map, axis=0)
    else:
        raise NotImplementedError
    
    return saliency_map
    
if __name__ == '__main__':
    if not os.path.exists('./sam_vit_b_01ec64.pth'):
        subprocess.run(['wget', 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'])
        
    from PIL import Image
    image = Image.open('ttt_heatmap.png')
    image = np.array(image)
    saliency_detect_mode = 1
    saliency_map = extract_saliency_map(image, mode=saliency_detect_mode)

    image_array = saliency_map * 255
    image_array = image_array.astype(np.uint8)
    Image.fromarray(image_array, 'L').save(f'{saliency_detect_mode}_saliency_map.jpg')  # 'L' mode for grayscale (black and white)

    images = [image, image]
    batch_saliency_map = extract_batch_saliency_map(images, mode=saliency_detect_mode)
    for i, saliency_map in enumerate(batch_saliency_map):
        image_array = saliency_map * 255
        image_array = image_array.astype(np.uint8)
        Image.fromarray(image_array, 'L').save(f'{i}_{saliency_detect_mode}_batch_saliency_map.png')
