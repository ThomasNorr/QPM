from argparse import ArgumentParser

import torch

#from sparsification.glm_saga import glm_saga
from sparsification import feature_helpers


def safe_zip(*args):
    for iterable in args[1:]:
        if len(iterable) != len(args[0]):
            print("Unequally sized iterables to zip, printing lengths")
            for i, entry in enumerate(args):
                print(i, len(entry))
            raise ValueError("Unequally sized iterables to zip")
    return zip(*args)


def compute_features_and_metadata(args, train_loader, test_loader, model, out_dir_feats, num_classes,
                                ):
    print("Computing/loading deep features...")

    Ntotal = len(train_loader.dataset)
    feature_loaders = {}
    # Compute Features for not augmented train and test set
    train_loader_transforms = train_loader.dataset.transform
    test_loader_transforms = test_loader.dataset.transform
    train_loader.dataset.transform = test_loader_transforms
    for mode, loader in zip(['train', 'test', ], [train_loader, test_loader, ]):  #
        print(f"For {mode} set...")

        sink_path = f"{out_dir_feats}/features_{mode}"
        metadata_path = f"{out_dir_feats}/metadata_{mode}.pth"

        feature_ds, feature_loader = feature_helpers.compute_features(loader,
                                                                      model,
                                                                      dataset_type=args.dataset_type,
                                                                      pooled_output=None,
                                                                      batch_size=args.batch_size,
                                                                      num_workers=0,  # args.num_workers,
                                                                      shuffle=(mode == 'test'),
                                                                      device=args.device,
                                                                      filename=sink_path, n_epoch=1,
                                                                      balance=False,
                                                                      )  # args.balance if mode == 'test' else False)

        if mode == 'train':
            metadata = feature_helpers.calculate_metadata(feature_loader,
                                                          num_classes=num_classes,
                                                          filename=metadata_path)
            if metadata["max_reg"]["group"] == 0.0:
                return None, False
            split_datasets, split_loaders = feature_helpers.split_dataset(feature_ds,
                                                                          Ntotal,
                                                                          val_frac=args.val_frac,
                                                                          batch_size=args.batch_size,
                                                                          num_workers=args.num_workers,
                                                                          random_seed=args.random_seed,
                                                                          shuffle=True,
                                                                          balance=False)
            feature_loaders.update({mm: add_index_to_dataloader(split_loaders[mi])
                                    for mi, mm in enumerate(['train', 'val'])})

        else:
            feature_loaders[mode] = feature_loader
    train_loader.dataset.transform = train_loader_transforms
    return feature_loaders, metadata

def get_feature_loaders(seed, log_folder,train_loader, test_loader, model, num_classes, ):
    args = get_default_args()
    args.random_seed = seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_folder = log_folder / "features"
    feature_loaders, metadata, = compute_features_and_metadata(args, train_loader, test_loader, model,
                                                               feature_folder
                                                               ,
                                                               num_classes,
                                                               )
    return feature_loaders, metadata, device,args
def add_index_to_dataloader(loader, sample_weight=None,):
    return torch.utils.data.DataLoader(
        IndexedDataset(loader.dataset, sample_weight=sample_weight),
        batch_size=loader.batch_size,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context
    )


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, ds, sample_weight=None):
        super(torch.utils.data.Dataset, self).__init__()
        self.dataset = ds
        self.sample_weight = sample_weight

    def __getitem__(self, index):
        val = self.dataset[index]
        if self.sample_weight is None:
            return val + (index,)
        else:
            weight = self.sample_weight[index]
            return val + (weight, index)

    def __len__(self):
        return len(self.dataset)


def get_default_args():
    # Default args from glm_saga, https://github.com/MadryLab/glm_saga
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--dataset-type', type=str, help='One of ["language", "vision"]')
    parser.add_argument('--dataset-path', type=str, help='path to dataset')
    parser.add_argument('--model-path', type=str, help='path to model checkpoint')
    parser.add_argument('--arch', type=str, help='model architecture type')
    parser.add_argument('--out-path', help='location for saving results')
    parser.add_argument('--cache', action='store_true', help='cache deep features')
    parser.add_argument('--balance', action='store_true', help='balance classes for evaluation')

    parser.add_argument('--device', default='cuda')
    parser.add_argument('--random-seed', default=0)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--val-frac', type=float, default=0.1)
    parser.add_argument('--lr-decay-factor', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.99)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--verbose', type=int, default=200)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--lookbehind', type=int, default=3)
    parser.add_argument('--lam-factor', type=float, default=0.001)
    parser.add_argument('--group', action='store_true')
    args = parser.parse_args()

    args = parser.parse_args()
    return args


def select_in_loader(feature_loaders, feature_selection):
    for dataset in feature_loaders["train"].dataset.dataset.dataset.datasets: # Val is indexed via the same dataset as train
        tensors = list(dataset.tensors)
        if tensors[0].shape[1] == len(feature_selection):
            continue
        tensors[0] = tensors[0][:, feature_selection]
        dataset.tensors = tensors
    for dataset in feature_loaders["test"].dataset.datasets:
        tensors = list(dataset.tensors)
        if tensors[0].shape[1] == len(feature_selection):
            continue
        tensors[0] = tensors[0][:, feature_selection]
        dataset.tensors = tensors
    return feature_loaders

