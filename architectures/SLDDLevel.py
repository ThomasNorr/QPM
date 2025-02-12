import torch.nn


class SLDDLevel(torch.nn.Module):
    def __init__(self, selection, weight_at_selection,mean, std, bias=None, retrain_normalisation=True):
        super().__init__()
        self.register_buffer('selection', torch.tensor(selection, dtype=torch.long))
        num_classes,        n_features = weight_at_selection.shape
        selected_mean = mean
        selected_std = std
        if len(selected_mean) != len(selection):
            selected_mean = selected_mean[selection]
            selected_std = selected_std[selection]
        self.mean = torch.nn.Parameter(selected_mean, requires_grad=retrain_normalisation)
        self.std = torch.nn.Parameter(selected_std, requires_grad=retrain_normalisation)
        if bias is not None:
            self.layer = torch.nn.Linear(n_features, num_classes)
            self.layer.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.layer = torch.nn.Linear(n_features, num_classes, bias=False)
        self.layer.weight = torch.nn.Parameter(weight_at_selection, requires_grad=False)

    @property
    def weight(self):
        return self.layer.weight

    @property
    def bias(self):
        if self.layer.bias is None:
            return torch.zeros(self.layer.out_features)
        else:
            return self.layer.bias


    def forward(self, input):
        input = (input - self.mean) / torch.clamp(self.std, min=1e-6)
        return self.layer(input), input
