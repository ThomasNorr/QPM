import numpy as np
import torch
from sklearn.cluster import MeanShift
from tqdm import tqdm

from evaluation.helpers import softmax_feature_maps


def compute_locality_bias(train_loader, model):
    loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    features = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _,  features_batch = model(input,
                                              with_final_features=True)
            features.append(features_batch.to('cpu'))
    features = torch.cat(features, dim=0)
    locality_bias =compute_average_softmax(features, loader, model, device)
    mask_bias =  compute_edge_mask(loader, model, device, features)
    normed_loc_bias = remove_outliers_and_scale(locality_bias)
    normed_mask_bias = remove_outliers_and_scale(mask_bias)
    feature_bias =normed_loc_bias * torch.tensor(normed_mask_bias >= 0).float().numpy() + normed_loc_bias.min() * torch.tensor(
    normed_mask_bias < 0).float().numpy()
    feature_bias = remove_outliers_and_scale(feature_bias)
    feature_bias = torch.tensor(feature_bias)
    return feature_bias

def get_assignments(features):
    features = np.array(features).reshape(-1, 1)
    clusterer = MeanShift()
    clusterer.fit(features)
    assignments = clusterer.labels_
    return assignments
def get_range_of_biggest_cluster(features_in):
    features = features_in.clone()
    assignments = get_assignments(features)
    uni = np.unique(assignments)
    max_size = -1
    for unique in uni:
        same = (assignments == unique).sum()
        if same > max_size:
            biggest = unique
            max_size = same

    high_end = features[assignments == biggest].max()
    low_end = features[assignments == biggest].min()
    return high_end, low_end


def remove_outliers_and_scale(feature_bias, scale= np.sqrt(10)*0.1):
    feature_bias = torch.tensor(feature_bias)
    high_end, low_end = get_range_of_biggest_cluster(feature_bias)
    # Clip all outliers from biggest clusters to extreme values
    feature_bias[feature_bias < low_end] = torch.abs(low_end) * torch.sign(
        feature_bias[feature_bias < low_end])
    feature_bias[feature_bias > high_end] = torch.abs(high_end) * torch.sign(
        feature_bias[feature_bias > high_end])
    # normalize to zero mean and max scale
    feature_bias = feature_bias - torch.mean(feature_bias)
    feature_bias = feature_bias / torch.max(torch.abs(feature_bias)) * scale
    return feature_bias.to("cpu").numpy()
def compute_average_softmax(features_train, loader, model, device):
    loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
    weights = (features_train / features_train.sum(dim=1, keepdim=True)).to(device)
    n_features = features_train.shape[1]
    answer = torch.zeros(n_features, device=device)
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.to(device)
            _, feature_maps = model(input, with_feature_maps=True,
                                              )
            softmaxed = softmax_feature_maps(feature_maps)
            max_map = torch.amax(softmaxed, dim=(2, 3))
            answer += torch.sum(max_map * weights[i * input.shape[0]: (i + 1) * input.shape[0]], dim=0)
    average_loc = answer / len(loader.dataset)
    return average_loc.cpu().numpy()

def check_on_edge(maps):
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    max_tensor = torch.stack((x, y), dim=2)
    is_max = max_tensor == maps.shape[2] - 1
    is_min = max_tensor == 0
    on_edge = torch.logical_or(is_max, is_min)
    on_edge = torch.any(on_edge, dim=2)
    return on_edge.float()

def compute_edge_mask(loader, model, device, features):
        loader = torch.utils.data.DataLoader(loader.dataset, batch_size=64, shuffle=False, num_workers=4)
        weights = (features / features.sum(dim=1, keepdim=True)).to(device)
        answer = torch.zeros(features.shape[1], device=device)
        no_scale = torch.zeros(features.shape[1], device=device)
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
                input = input.to(device)
                _, feature_maps, features = model(input, with_feature_maps=True,
                                                  with_final_features=True)
                diff_to_middle = check_on_edge(feature_maps)
                no_scale += torch.sum(diff_to_middle, dim=0)
                answer += torch.sum(diff_to_middle * weights[i * 64:(i + 1) * 64], dim=0)
        return -answer.cpu().numpy()


def center_bias(loader, features, model):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = (features / features.sum(dim=1, keepdim=True)).to(device)
    answer = torch.zeros(features.shape[1], device=device)
    no_scale = torch.zeros(features.shape[1], device=device)
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(loader), total=len(loader)):
            input = input.to(device)
            _, feature_maps, features = model(input, with_feature_maps=True,
                                              with_image=False,
                                              with_final_features=True)
            diff_to_middle = 1 / (diff_to_edge(feature_maps) + 1)
            no_scale += torch.sum(diff_to_middle, dim=0)
            answer += torch.sum(diff_to_middle * weights[i * 64:(i + 1) * 64], dim=0)
            #  flat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[1], -1)

    return -answer.cpu().numpy()


def diff_to_edge(maps):
    argmaxes = torch.argmax(maps.flatten(-2), dim=(2))
    x, y = torch.div(argmaxes, maps.shape[2], rounding_mode='trunc'), torch.remainder(argmaxes, maps.shape[2])
    right_edge = torch.abs(x - (maps.shape[2] - 1))
    left_edge = x
    top_edge = y
    bottom_edge = torch.abs(y - (maps.shape[2] - 1))
    diffs = torch.stack((right_edge, left_edge, top_edge, bottom_edge), dim=2)
    diffs = torch.min(diffs, dim=2)[0]
    return diffs