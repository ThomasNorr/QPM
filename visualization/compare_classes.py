from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.hipify.hipify_python import mapping

from configs.dataset_params import normalize_params
from dataset_classes.cub200 import CUB200Class, load_cub_class_mapping
from evaluation.load_model import load_model
from get_data import get_data
from visualization.get_heatmaps import get_visualizations
from configs.dataset_params import normalize_params
from visualization.pairstoViz import find_easier_interpretable_pairs, select_clearly_activating_separable_samples


def get_combined_indices(combined_indices):
    # Returns feature indices joined, so that first uniques of class 1, then shared, then uniques of class two.
    rel_values = list(combined_indices.values())
    shared_indices = set(rel_values[0]).intersection(rel_values[1])
    middle_indices = sorted(list(shared_indices))
    unique_first = [i for i in rel_values[0] if i not in shared_indices]
    unique_second = [i for i in rel_values[1] if i not in shared_indices]
    total_indices = unique_first + middle_indices + unique_second
    return total_indices

def viz_model(model,data_loader, class_indices ,test_key, gamma = 3, norm_across_channels = True, size=(2.5,2.5)):
    images = []
    image_unnormalized = []
    data_mean, data_std = normalize_params[data_loader.dataset.name]["mean"], normalize_params[data_loader.dataset.name]["std"]
    assert len(class_indices) == 2
    class_names = None
    if isinstance(data_loader.dataset, CUB200Class):
        mapping = load_cub_class_mapping()
        class_names = [mapping[str(x)] for x in class_indices]
    combined_indices = {}
    for j, c_index in enumerate(class_indices):
        rel_indices = data_loader.dataset.get_indices_for_target(c_index)
        class_features = model.linear.weight[c_index].nonzero().flatten().tolist()
        combined_indices[c_index] = class_features
        class_images = []
        for idx in rel_indices:
            image, label = data_loader.dataset[idx]
            assert label == c_index
            class_images.append(image)
        image = select_clearly_activating_separable_samples(model, class_images, c_index)
        image_unnormalized.append(image* data_std[:, None, None] + data_mean[:, None, None])

        images.append(image)
    combined_indices = get_combined_indices(combined_indices)
    img_full = torch.stack(images)
    img_full = img_full.to("cuda")
    image_unnormalized = torch.stack(image_unnormalized)
    visualizations = get_visualizations(combined_indices, img_full,image_unnormalized, model, gamma=gamma,
                                               norm_across_images=norm_across_channels, )
    fig, axes  = plt.subplots(2, len(visualizations) + 1, figsize=(size[0]*(len(visualizations) +1), size[1] * 2))

    for i, img in enumerate(image_unnormalized):
        ax = axes[i,0]
        ax.imshow(img.permute(1, 2, 0))
        if class_names is not None:
            ax.set_ylabel(class_names[i])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    for i, feat_maps_for_samples in enumerate(visualizations):
        for j, feat_for_sample_viz in enumerate(feat_maps_for_samples):
            ax = axes[j, i+1]
            ax.imshow(feat_for_sample_viz.permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    viz_folder = Path.home() / "tmp" / f"{test_key}vizQPMClassComparisons"
    viz_folder.mkdir(exist_ok=True, parents=True)
    plt.savefig(viz_folder / f"{'_'.join([str(x) for x in class_indices])}.png",  bbox_inches='tight')



if  __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="CUB2011", type=str, help='dataset name',
                        choices=["CUB2011", "ImageNet", "TravelingBirds", "StanfordCars"])
    parser.add_argument('--arch', default="resnet50", type=str, help='Backbone Feature Extractor',
                        choices=["resnet50", "resnet18"])
    parser.add_argument('--model_type', default="qpm", type=str, help='Type of Model', choices=["qsenn", "sldd", "qpm"])
    parser.add_argument('--seed', default=504405, type=int, # 504405 is good
                        help='seed, used for naming the folder and random processes. Could be useful to set to have multiple finetune runs (e.g. Q-SENN and SLDD) on the same dense model')  # 769567, 552629
    parser.add_argument('--cropGT', default=True, type=bool,
                        help='Whether to crop CUB/TravelingBirds based on GT Boundaries')
    parser.add_argument('--n_features', default=50, type=int, help='How many features to select')  # 769567
    parser.add_argument('--n_per_class', default=5, type=int, help='How many features to assign to each class')
    parser.add_argument('--img_size', default=224, type=int, help='Image size')
    parser.add_argument('--reduced_strides', default=True, type=bool,
                        help='Whether to use reduced strides for resnets')
    parser.add_argument("--folder", default=None, type=str, help="Folder to load model from")
    args = parser.parse_args()
    train_loader, test_loader = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    model = load_model(args.dataset, args.arch, args.seed, args.model_type, args.cropGT, args.n_features,
                       args.n_per_class, args.img_size, args.reduced_strides, args.folder)
    interesting_pairs = find_easier_interpretable_pairs(model, train_loader,min_sim= 0.8)
    if args.dataset == "CUB2011":
        mapping = load_cub_class_mapping()
        class_names = [(mapping[str(x)], mapping[str(y)]) for x, y in interesting_pairs]
    train_loader.dataset.transform = test_loader.dataset.transform
    for pair in interesting_pairs:
        class_indices = list(pair)
        viz_model(model, test_loader, class_indices, "Test")

        viz_model(model, train_loader, class_indices, "Train")