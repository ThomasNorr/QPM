import math
import os
import sys

import torch.cuda

import sparsification.utils

sys.path.append('')
import numpy as np
import torch as ch
from torch.utils.data import Subset
from tqdm import tqdm



# From glm_saga
def get_features_batch(batch, model,  device='cuda'):
    if not torch.cuda.is_available():
        device = "cpu"
    ims, targets = batch
    output, latents = model(ims.to(device), with_final_features=True )
    return latents, targets


def compute_features(loader, model, dataset_type, pooled_output,
                     batch_size, num_workers,
                     shuffle=False, device='cpu', n_epoch=1,
                     filename=None, chunk_threshold=20000, balance=False):
    """Compute deep features for a given dataset using a modeln and returnss
    them as a pytorch dataset and loader.
    Args:
        loader : Torch data loader
        model: Torch model
        dataset_type (str): One of vision or language
        pooled_output (bool): Whether or not to pool outputs
          (only relevant for some language models)
        batch_size (int): Batch size for output loader
        num_workers (int): Number of workers to use for output loader
        shuffle (bool): Whether or not to shuffle output data loaoder
        device (str): Device on which to keep the model
        filename (str):Optional file to cache computed feature. Recommended
            for large dataset_classes like ImageNet.
        chunk_threshold (int): Size of shard while caching
        balance (bool): Whether or not to balance output data loader
            (only relevant for some language models)
    Returns:
        feature_dataset: Torch dataset with deep features
        feature_loader: Torch data loader with deep features
    """
    if torch.cuda.is_available():
        device = "cuda"
        print("mem_get_info before", torch.cuda.mem_get_info())
        torch.cuda.empty_cache()
        print("mem_get_info after", torch.cuda.mem_get_info())
        model = model.to(device)
    if filename is None or not os.path.exists(os.path.join(filename, f'0_features.npy')):
        model.eval()
        all_latents, all_targets, all_images = [], [], []
        Nsamples, chunk_id = 0, 0
        for idx_epoch in range(n_epoch):
            for batch_idx, batch in tqdm(enumerate(loader), total=len(loader)):
                with ch.no_grad():
                    latents, targets = get_features_batch(batch, model,
                                                          device=device)
                if batch_idx == 0:
                    print("Latents shape", latents.shape)
                Nsamples += latents.size(0)

                all_latents.append(latents.cpu())
                if len(targets.shape) > 1:
                    targets = targets[:, 0]
                all_targets.append(targets.cpu())
                # all_images.append(batch[0])
                if filename is not None and Nsamples > chunk_threshold:
                    if not os.path.exists(filename): os.makedirs(filename)
                    np.save(os.path.join(filename, f'{chunk_id}_features.npy'), ch.cat(all_latents).numpy())
                    np.save(os.path.join(filename, f'{chunk_id}_labels.npy'), ch.cat(all_targets).numpy())

                    all_latents, all_targets, Nsamples = [], [], 0
                    chunk_id += 1

        if filename is not None and Nsamples > 0:
            if not os.path.exists(filename): os.makedirs(filename)
            np.save(os.path.join(filename, f'{chunk_id}_features.npy'), ch.cat(all_latents).numpy())
            np.save(os.path.join(filename, f'{chunk_id}_labels.npy'), ch.cat(all_targets).numpy())
        #  np.save(os.path.join(filename, f'{chunk_id}_images.npy'), ch.cat(all_images).numpy())
    feature_dataset = load_features(filename) if filename is not None else \
        ch.utils.data.TensorDataset(ch.cat(all_latents), ch.cat(all_targets))
    if balance:
        feature_dataset = balance_dataset(feature_dataset)

    feature_loader = ch.utils.data.DataLoader(feature_dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=shuffle)

    return feature_dataset, feature_loader


def load_feature_loader(out_dir_feats, val_frac, batch_size, num_workers, random_seed):
    feature_loaders = {}
    for mode in ['train', 'test']:
        print(f"For {mode} set...")
        sink_path = f"{out_dir_feats}/features_{mode}"
        metadata_path = f"{out_dir_feats}/metadata_{mode}.pth"
        feature_ds = load_features(sink_path)
        feature_loader = ch.utils.data.DataLoader(feature_ds,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size)
        if mode == 'train':
            metadata = calculate_metadata(feature_loader,
                                          num_classes=2048,
                                          filename=metadata_path)
            split_datasets, split_loaders = split_dataset(feature_ds,
                                                          len(feature_ds),
                                                          val_frac=val_frac,
                                                          batch_size=batch_size,
                                                          num_workers=num_workers,
                                                          random_seed=random_seed,
                                                          shuffle=True)
            feature_loaders.update({mm: sparsification.utils.add_index_to_dataloader(split_loaders[mi])
                                    for mi, mm in enumerate(['train', 'val'])})

        else:
            feature_loaders[mode] = feature_loader
    return feature_loaders, metadata


def balance_dataset(dataset):
    """Balances a given dataset to have the same number of samples/class.
    Args:
        dataset : Torch dataset
    Returns:
        Torch dataset with equal number of samples/class
    """

    print("Balancing dataset...")
    n = len(dataset)
    labels = ch.Tensor([dataset[i][1] for i in range(n)]).int()
    n0 = sum(labels).item()
    I_pos = labels == 1

    idx = ch.arange(n)
    idx_pos = idx[I_pos]
    ch.manual_seed(0)
    I = ch.randperm(n - n0)[:n0]
    idx_neg = idx[~I_pos][I]
    idx_bal = ch.cat([idx_pos, idx_neg], dim=0)
    return Subset(dataset, idx_bal)


def load_metadata(feature_path):
    return ch.load(os.path.join(feature_path, f'metadata_train.pth'))


def get_mean_std(feature_path):
    metadata = load_metadata(feature_path)
    return metadata["X"]["mean"], metadata["X"]["std"]


def load_features_dataset_mode(feature_path, mode='test',
                               num_workers=10, batch_size=128):
    """Loads precomputed deep features corresponding to the
    train/test set along with normalization statitic.
    Args:
        feature_path (str): Path to precomputed deep features
        mode (str): One of train or tesst
        num_workers (int): Number of workers to use for output loader
        batch_size (int): Batch size for output loader

    Returns:
        features (np.array): Recovered deep features
        feature_mean: Mean of deep features
        feature_std: Standard deviation of deep features
    """
    feature_dataset = load_features(os.path.join(feature_path, f'features_{mode}'))
    feature_loader = ch.utils.data.DataLoader(feature_dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=False)
    feature_metadata = ch.load(os.path.join(feature_path, f'metadata_train.pth'))
    feature_mean, feature_std = feature_metadata['X']['mean'], feature_metadata['X']['std']
    return feature_loader, feature_mean, feature_std


def load_joint_dataset(feature_path, mode='test',
                       num_workers=10, batch_size=128):
    feature_dataset = load_features(os.path.join(feature_path, f'features_{mode}'))
    feature_loader = ch.utils.data.DataLoader(feature_dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=False)
    features = []
    labels = []
    for _, (feature, label) in tqdm(enumerate(feature_loader), total=len(feature_loader)):
        features.append(feature)
        labels.append(label)
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    dataset = ch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(labels))
    return dataset


def load_features_mode(feature_path, mode='test',
                       num_workers=10, batch_size=128):
    """Loads precomputed deep features corresponding to the
    train/test set along with normalization statitic.
    Args:
        feature_path (str): Path to precomputed deep features
        mode (str): One of train or tesst
        num_workers (int): Number of workers to use for output loader
        batch_size (int): Batch size for output loader

    Returns:
        features (np.array): Recovered deep features
        feature_mean: Mean of deep features
        feature_std: Standard deviation of deep features
    """
    feature_dataset = load_features(os.path.join(feature_path, f'features_{mode}'))
    feature_loader = ch.utils.data.DataLoader(feature_dataset,
                                              num_workers=num_workers,
                                              batch_size=batch_size,
                                              shuffle=False)

    feature_metadata = ch.load(os.path.join(feature_path, f'metadata_train.pth'))
    feature_mean, feature_std = feature_metadata['X']['mean'], feature_metadata['X']['std']

    features = []

    for _, (feature, _) in tqdm(enumerate(feature_loader), total=len(feature_loader)):
        features.append(feature)

    features = ch.cat(features).numpy()
    return features, feature_mean, feature_std


def load_features(feature_path):
    """Loads precomputed deep features.
    Args:
        feature_path (str): Path to precomputed deep features

    Returns:
        Torch dataset with recovered deep features.
    """
    if not os.path.exists(os.path.join(feature_path, f"0_features.npy")):
        raise ValueError(f"The provided location {feature_path} does not contain any representation files")

    ds_list, chunk_id = [], 0
    while os.path.exists(os.path.join(feature_path, f"{chunk_id}_features.npy")):
        features = ch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_features.npy"))).float()
        labels = ch.from_numpy(np.load(os.path.join(feature_path, f"{chunk_id}_labels.npy"))).long()
        ds_list.append(ch.utils.data.TensorDataset(features, labels))
        chunk_id += 1

    print(f"==> loaded {chunk_id} files of representations...")
    return ch.utils.data.ConcatDataset(ds_list)


def calculate_metadata(loader, num_classes=None, filename=None):
    """Calculates mean and standard deviation of the deep features over
    a given set of images.
    Args:
        loader : torch data loader
        num_classes (int): Number of classes in the dataset
        filename (str): Optional filepath to cache metadata. Recommended
            for large dataset_classes like ImageNet.

    Returns:
        metadata (dict): Dictionary with desired statistics.
    """

    if filename is not None and os.path.exists(filename):
        print("loading Metadata from ", filename)
        return ch.load(filename)

    # Calculate number of classes if not given
    if num_classes is None:
        num_classes = 1
        for batch in loader:
            y = batch[1]
            print(y)
            num_classes = max(num_classes, y.max().item() + 1)

    eye = ch.eye(num_classes)

    X_bar, y_bar, y_max, n = 0, 0, 0, 0

    # calculate means and maximum
    print("Calculating means")
    for ans in tqdm(loader, total=len(loader)):
        X, y = ans[:2]
        X_bar += X.sum(0)
        y_bar += eye[y].sum(0)
        y_max = max(y_max, y.max())
        n += y.size(0)
    X_bar = X_bar.float() / n
    y_bar = y_bar.float() / n

    # calculate std
    X_std, y_std = 0, 0
    print("Calculating standard deviations")
    for ans in tqdm(loader, total=len(loader)):
        X, y = ans[:2]
        X_std += ((X - X_bar) ** 2).sum(0)
        y_std += ((eye[y] - y_bar) ** 2).sum(0)
    X_std = ch.sqrt(X_std.float() / n)
    y_std = ch.sqrt(y_std.float() / n)

    # calculate maximum regularization
    inner_products = 0
    print("Calculating maximum lambda")
    for ans in tqdm(loader, total=len(loader)):
        X, y = ans[:2]
        y_map = (eye[y] - y_bar) / y_std
        inner_products += X.t().mm(y_map) * y_std

    inner_products_group = inner_products.norm(p=2, dim=1)

    metadata = {
        "X": {
            "mean": X_bar,
            "std": X_std,
            "num_features": X.size()[1:],
            "num_examples": n
        },
        "y": {
            "mean": y_bar,
            "std": y_std,
            "num_classes": y_max + 1
        },
        "max_reg": {
            "group": inner_products_group.abs().max().item() / n,
            "nongrouped": inner_products.abs().max().item() / n
        }
    }

    if filename is not None:
        ch.save(metadata, filename)

    return metadata


def split_dataset(dataset, Ntotal, val_frac,
                  batch_size, num_workers,
                  random_seed=0, shuffle=True, balance=False):
    """Splits a given dataset into train and validation
    Args:
        dataset : Torch dataset
        Ntotal: Total number of dataset samples
        val_frac: Fraction to reserve for validation
        batch_size (int): Batch size for output loader
        num_workers (int): Number of workers to use for output loader
        random_seed (int): Random seed
        shuffle (bool): Whether or not to shuffle output data loaoder
        balance (bool): Whether or not to balance output data loader
            (only relevant for some language models)

    Returns:
        split_datasets (list): List of dataset_classes (one each for train and val)
        split_loaders (list): List of loaders (one each for train and val)
    """

    Nval = math.floor(Ntotal * val_frac)
    train_ds, val_ds = ch.utils.data.random_split(dataset,
                                                  [Ntotal - Nval, Nval],
                                                  generator=ch.Generator().manual_seed(random_seed))
    if balance:
        val_ds = balance_dataset(val_ds)
    split_datasets = [train_ds, val_ds]

    split_loaders = []
    for ds in split_datasets:
        split_loaders.append(ch.utils.data.DataLoader(ds,
                                                      num_workers=num_workers,
                                                      batch_size=batch_size,
                                                      shuffle=shuffle))
    return split_datasets, split_loaders
