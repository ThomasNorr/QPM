
import torch


class NormalizedRepresentation(torch.nn.Module):
    def __init__(self, loader, metadata, device='cuda', tol=1e-5):
        super(NormalizedRepresentation, self).__init__()

        assert metadata is not None
        self.device = device
        self.mu = metadata['X']['mean']
        self.sigma = torch.clamp(metadata['X']['std'], tol)

    def forward(self, X):
        return (X - self.mu.to(self.device)) / self.sigma.to(self.device)

