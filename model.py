import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=15):
    # Load lightweight MobileNetV2 pre-trained on ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # Freeze pre-trained feature extractor layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final classification head for our 15 plant disease classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    net = build_model(num_classes=15)
    print("Model Architecture Initialized Successfully!")
    print(net)