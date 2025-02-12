import torch
from torch import nn

from architectures.SLDDLevel import SLDDLevel


class FinalLayer():
    def __init__(self, num_classes,  n_features):
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(n_features, num_classes)
        self.featureDropout = torch.nn.Dropout(0.2)
        self.selection = None

    def transform_output(self,  feature_maps, with_feature_maps,
                         with_final_features):
        if self.selection is not None:
            feature_maps = feature_maps[:, self.selection]
        x = self.avgpool(feature_maps)
        pre_out = torch.flatten(x, 1)
        final_features = self.featureDropout(pre_out)
        final = self.linear(final_features)
        if isinstance(final, tuple): # In case of SLDD, features are also returned, as they are scaled (initially for normalisation)
            final, final_features = final
        final = [final]
        if with_feature_maps:
            final.append(feature_maps)
        if with_final_features:
            final.append(final_features)
        if len(final) == 1:
            final = final[0]
        return final


    def set_model_sldd(self, selection, weight, mean, std, bias = None, retrain_normalisation = True):
        self.selection = selection
        self.linear = SLDDLevel(selection, weight, mean, std, bias, retrain_normalisation)
        self.featureDropout = torch.nn.Dropout(0.1)