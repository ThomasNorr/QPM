import torch
from tqdm import tqdm

from training.utils import VariableLossLogPrinter


def get_acc(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return correct / total * 100



def train(model, train_loader, optimizer, fdl, epoch):
    model.train()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    VariableLossPrinter = VariableLossLogPrinter()
    model = model.to(device)
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in iterator:
        on_device = data.to(device)
        target_on_device = target.to(device)

        output, feature_maps = model(on_device, with_feature_maps=True)
        loss = torch.nn.functional.cross_entropy(output, target_on_device)

        fdl_loss = fdl(feature_maps, output)
        total_loss = loss + fdl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        acc = get_acc(output, target_on_device)
        VariableLossPrinter.log_loss("Train Acc", acc, on_device.size(0))
        VariableLossPrinter.log_loss("CE-Loss", loss.item(), on_device.size(0))
        VariableLossPrinter.log_loss("FDL", fdl_loss.item(), on_device.size(0))
        VariableLossPrinter.log_loss("Total-Loss", total_loss.item(), on_device.size(0))
        iterator.set_description(f"Train Epoch:{epoch} Metrics: {VariableLossPrinter.get_loss_string()}")
    print("Trained model for one epoch ",  epoch," with lr group 0: ", optimizer.param_groups[0]["lr"])
    return model


def test(model, test_loader, epoch):
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    VariableLossPrinter = VariableLossLogPrinter()
    iterator = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for batch_idx, (data, target) in iterator:
            on_device = data.to(device)
            target_on_device = target.to(device)
            output, feature_maps = model(on_device, with_feature_maps=True)
            loss = torch.nn.functional.cross_entropy(output, target_on_device)
            acc = get_acc(output, target_on_device)
            VariableLossPrinter.log_loss("Test Acc", acc, on_device.size(0))
            VariableLossPrinter.log_loss("CE-Loss", loss.item(), on_device.size(0))
            iterator.set_description(f"Test Epoch:{epoch} Metrics: {VariableLossPrinter.get_loss_string()}")
