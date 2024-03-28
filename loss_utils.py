import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision  import transforms

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_submodel = nn.Sequential(*list(vgg)[:23])  # Using up to the 4th block before maxpool
        for param in self.vgg_submodel.parameters():
            param.requires_grad = False  # Freeze the model
        self.tranformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                ])
    def forward(self, input_image, output_image):
        input_features = self.vgg_submodel(self.tranformation(input_image))
        output_features = self.vgg_submodel(self.tranformation(output_image))
        return nn.functional.mse_loss(input_features, output_features)

