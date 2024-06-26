import torch
import torch.nn as nn
from torchvision.models import vgg16
from torchvision  import transforms
from transformers import CLIPProcessor, CLIPModel

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features
        self.vgg_submodel = nn.Sequential(*list(vgg)[:23]) 
        for param in self.vgg_submodel.parameters():
            param.requires_grad = False  # Freeze the model
        self.tranformation = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225]),
                                                ])
    def forward(self, input_image, output_image):
        input_image = input_image.float()
        output_image = output_image.float()
        input_features = self.vgg_submodel(self.tranformation(input_image))
        output_features = self.vgg_submodel(self.tranformation(output_image))
        return nn.functional.mse_loss(input_features, output_features)

class ClipSimilarityLoss():
    def __init__(self,text_description):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.text_description = text_description

    def __call__(self,image):
        inputs = self.clip_processor(text=[self.text_description], images=image, return_tensors="pt", padding=True)
        outputs = self.clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        print(logits)
        similarity = logits_per_image.flatten()
        loss = -similarity.mean()
        return loss