from architectures.resnet import resnet50, resnet34, resnet18


def get_model(arch, num_classes, changed_strides=True):
    if arch == "resnet50":
        model = resnet50(True, num_classes=num_classes, changed_strides=changed_strides)
    elif arch == "resnet34":
        model = resnet34(True, num_classes=num_classes, changed_strides=changed_strides)
    elif arch == "resnet18":
        model = resnet18(True, num_classes=num_classes, changed_strides=changed_strides)
    return  model