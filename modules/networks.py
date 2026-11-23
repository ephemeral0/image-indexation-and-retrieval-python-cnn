
import torch
import torchvision.models as models


def load_network(network_name):

    if network_name == "ResNet-18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif network_name == "ResNet-50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model = torch.nn.Sequential(*list(model.children())[:-1])

    elif network_name == "VGG-16":
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
        model.classifier = torch.nn.Identity()

    else:
        raise ValueError("Unknown network: " + network_name)

    model.eval()

    print(f"{network_name} loaded successfully!")

    return model
