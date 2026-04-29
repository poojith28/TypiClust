import os

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEP_AL_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
_PROJECT_ROOT = os.path.abspath(os.path.join(_DEEP_AL_ROOT, '..'))

DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'../../scan/results/cifar-10/pretext/features_seed1.npy',
            'CIFAR100':'../../scan/results/cifar-100/pretext/features_seed1.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed1.npy',
            'IMAGENET50': '../../dino/runs/trainfeat.pth',
            'IMAGENET100': '../../dino/runs/trainfeat.pth',
            'IMAGENET200': '../../dino/runs/trainfeat.pth',
        },
    'test':
        {
            'CIFAR10': '../../scan/results/cifar-10/pretext/test_features_seed1.npy',
            'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed1.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed1.npy',
            'IMAGENET50': '../../dino/runs/testfeat.pth',
            'IMAGENET100': '../../dino/runs/testfeat.pth',
            'IMAGENET200': '../../dino/runs/testfeat.pth',
        }
}


def _candidate_feature_paths(raw_path):
    candidates = []

    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        candidates.append(os.path.abspath(os.path.join(_THIS_DIR, raw_path)))
        candidates.append(os.path.abspath(os.path.join(_DEEP_AL_ROOT, raw_path)))
        if raw_path.startswith('../../'):
            candidates.append(os.path.abspath(os.path.join(_PROJECT_ROOT, raw_path[6:])))

    normalized = []
    seen = set()
    for path in candidates:
        norm = os.path.normpath(path)
        if norm not in seen:
            normalized.append(norm)
            seen.add(norm)
    return normalized


def load_features(ds_name, seed=1, train=True, normalized=True):
    " load pretrained features for a dataset "
    split = "train" if train else "test"
    raw_path = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)
    candidate_paths = _candidate_feature_paths(raw_path)
    fname = next((path for path in candidate_paths if os.path.exists(path)), None)
    if fname is None:
        raise FileNotFoundError(
            "Could not find pretrained features for dataset={} split={} seed={}."
            " Tried: {}".format(ds_name, split, seed, candidate_paths)
        )
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features

