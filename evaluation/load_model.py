from argparse import ArgumentParser
from pathlib import Path

import torch

from architectures.model_mapping import get_model
from configs.dataset_params import dataset_constants
from evaluation.qpm_metrics import eval_model_on_all_qpm_metrics
from evaluation.qsenn_metrics import eval_model_on_all_qsenn_metrics
from get_data import get_data


def extract_sel_mean_std_bias_assignemnt(state_dict):
    feature_sel = state_dict["linear.selection"]
    weight_at_selection = state_dict["linear.layer.weight"]
    mean = state_dict["linear.mean"]
    std = state_dict["linear.std"]
    if "linear.linear.bias" in state_dict:
        bias = state_dict["linear.linear.bias"]
    else:
        bias = None
    return feature_sel, weight_at_selection, mean, std, bias


def eval_model(model, train_loader, test_loader):
  #  metrics_finetuned = eval_model_on_all_qsenn_metrics(model, test_loader, train_loader)
    metrics_finetuned = eval_model_on_all_qpm_metrics(model, test_loader, train_loader)
    print(metrics_finetuned)

def load_model(dataset, arch,seed=None, model_type="qsenn",crop = True, n_features = 50, n_per_class=5, img_size=448, reduced_strides=False, folder = None):
    n_classes = dataset_constants[dataset]["num_classes"]

    model = get_model(arch, n_classes, reduced_strides)
    if folder is None:
        if crop: dataset += "_crop"
        folder = Path.home() / f"tmp/{arch}/{dataset}/{seed}/"
    state_dict = torch.load(folder / f"{model_type}_{n_features}_{n_per_class}_FinetunedModel.pth")
    feature_sel, sparse_layer, current_mean, current_std, bias_sparse = extract_sel_mean_std_bias_assignemnt(state_dict)
    model.set_model_sldd(feature_sel, sparse_layer, current_mean, current_std, bias_sparse)
    model.load_state_dict(state_dict)
    model.eval()
    return model



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="CUB2011", type=str, help='dataset name', choices=["CUB2011", "ImageNet", "TravelingBirds", "StanfordCars"])
    parser.add_argument('--arch', default="resnet50", type=str, help='Backbone Feature Extractor', choices=["resnet50", "resnet18"])
    parser.add_argument('--model_type', default="qpm", type=str, help='Type of Model', choices=["qsenn", "sldd", "qpm"])
    parser.add_argument('--seed', default=504405, type=int, help='seed, used for naming the folder and random processes. Could be useful to set to have multiple finetune runs (e.g. Q-SENN and SLDD) on the same dense model') # 769567, 552629
    parser.add_argument('--cropGT', default=True, type=bool,
                        help='Whether to crop CUB/TravelingBirds based on GT Boundaries')
    parser.add_argument('--n_features', default=50, type=int, help='How many features to select')
    parser.add_argument('--n_per_class', default=5, type=int, help='How many features to assign to each class')
    parser.add_argument('--img_size', default=224, type=int, help='Image size')
    parser.add_argument('--reduced_strides', default=True, type=bool, help='Whether to use reduced strides for resnets')
    parser.add_argument("--folder", default=None, type=str, help="Folder to load model from. If not given, it will be generated based on the other parameters")
    args = parser.parse_args()
    train_loader, test_loader = get_data(args.dataset, crop=args.cropGT, img_size=args.img_size)
    model = load_model(args.dataset, args.arch, args.seed, args.model_type,args.cropGT,  args.n_features, args.n_per_class, args.img_size, args.reduced_strides, args.folder)
    eval_model(model,train_loader, test_loader)