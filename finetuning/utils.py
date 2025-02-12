from FeatureDiversityLoss import FeatureDiversityLoss
from train import train, test
from training.optim import get_optimizer


def train_n_epochs(model, beta,optimization_schedule, train_loader, test_loader):
    optimizer, schedule, epochs = get_optimizer(model, optimization_schedule)
    fdl = FeatureDiversityLoss(beta, model.linear)
    for epoch in range(epochs):
        model = train(model, train_loader, optimizer, fdl, epoch)
        schedule.step()
        if epoch % 5 == 0 or epoch+1 == epochs:
            test(model, test_loader, epoch)
    return model