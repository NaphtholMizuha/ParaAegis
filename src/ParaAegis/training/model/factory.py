from .cnn import Cnn, CnnGray
from torchvision.models import resnet18, shufflenet_v2_x1_0, vit_b_16
import timm
def fetch_model(name: str, **kwargs):
    if name == "cnn":
        return Cnn(**kwargs)
    elif name == "cnn-gray":
        return CnnGray(**kwargs)
    elif name == "resnet":
        return resnet18(num_classes=10, **kwargs)
    elif name == "shufflenet":
        return shufflenet_v2_x1_0(num_classes=10, **kwargs)
    elif name == "vit":
        return vit_b_16(num_classes=10)
    elif name == "mobilevit":
        return timm.create_model('mobilevitv2_050', pretrained=False, num_classes=10)
    elif name == "levit":
        return timm.create_model('levit_128', pretrained=False, num_classes=10)
    else:
        raise ValueError(f"Unknown model name: {name}")