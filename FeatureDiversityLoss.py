import torch
from torch import nn

"""
Feature Diversity Loss:
Usage to replicate paper:
Call 
loss_function = FeatureDiversityLoss(0.196, linear) 
to inititalize loss with linear layer of model.
At each mini batch get feature maps (Output of final convolutional layer) and add to Loss:
loss += loss_function(feature_maps, outputs)
"""


class FeatureDiversityLoss(nn.Module):
    def __init__(self, scaling_factor, linear):
        super().__init__()
        self.scaling_factor = scaling_factor #* 0
        print("Scaling Factor: ", self.scaling_factor)
        self.linearLayer = linear

    def initialize(self, linearLayer):
        self.linearLayer = linearLayer

    def get_weights(self, outputs):
        weight_matrix = self.linearLayer.weight
        weight_matrix = torch.abs(weight_matrix)
        top_classes = torch.argmax(outputs, dim=1)
        relevant_weights = weight_matrix[top_classes]
        return relevant_weights

    def forward(self, feature_maps, outputs):
        relevant_weights = self.get_weights(outputs)
        relevant_weights = norm_vector(relevant_weights)
        feature_maps = preserve_avg_func(feature_maps)
        flattened_feature_maps = feature_maps.flatten(2)
        batch, features, map_size = flattened_feature_maps.size()
        relevant_feature_maps = flattened_feature_maps * relevant_weights[..., None]
        diversity_loss = torch.sum(
            torch.amax(relevant_feature_maps, dim=1))
        return -diversity_loss / batch * self.scaling_factor


def norm_vector(x):
    return x / (torch.norm(x, dim=1) + 1e-5)[:, None]


def preserve_avg_func(x):
    avgs = torch.mean(x, dim=[2, 3])
    max_avgs = torch.max(avgs, dim=1)[0]
    scaling_factor = avgs / torch.clamp(max_avgs[..., None], min=1e-6)
    softmaxed_maps = softmax_feature_maps(x)
    scaled_maps = softmaxed_maps * scaling_factor[..., None, None]
    return scaled_maps


def softmax_feature_maps(x):
    return torch.softmax(x.reshape(x.size(0), x.size(1), -1), 2).view_as(x)

