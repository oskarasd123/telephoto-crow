import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from dataloader import MergedImageFolder
import numpy as np
from torchvision import transforms
import cv2
import random

device = "cuda"


class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, mid_ratio : float, kernel_size : int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.mid_channels = int((in_channels + out_channels) * mid_ratio / 2)
        assert kernel_size % 2 == 1, f"kernel size must be odd. got {kernel_size}"
        self.up_conv = nn.Conv2d(in_channels, self.mid_channels, kernel_size, 1, (kernel_size-1)//2)
        self.down_conv = nn.Conv2d(self.mid_channels, out_channels, 1, 1, 0)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.output_norm = nn.GroupNorm(out_channels, out_channels)
    
    def forward(self, x):
        out = self.residual_conv(x) + self.down_conv(F.gelu(self.up_conv((x))))
        return self.output_norm(out)


class ConvBlockDownSample(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, mid_ratio : float, kernel_size : int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.mid_channels = int((in_channels + out_channels) * mid_ratio / 2)
        assert kernel_size%2==1, f"kernel size must be odd. got {kernel_size}"
        self.up_conv = nn.Conv2d(in_channels, self.mid_channels, kernel_size, 1, (kernel_size-1)//2)
        self.down_conv = nn.Conv2d(self.mid_channels, out_channels, 4, 2, 1) # down samples by 2x
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 2, 2, 0) # also down samples by 2x, but uses 2x2 kernel
        self.output_norm = nn.GroupNorm(out_channels, out_channels)
    
    def forward(self, x):
        assert x.size(2) % 2 == 0 and x.size(3) % 2 == 0, f"got input shape: {x.shape}. This module doesn't handle padding for uneven inputs"
        out = self.residual_conv(x) + self.down_conv(F.gelu(self.up_conv((x))))
        return self.output_norm(out)


class Detector(nn.Module):
    def __init__(self, in_channels : int, out_channels : int):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlockDownSample(in_channels, 32, 2),
            ConvBlockDownSample(32, 64, 2),
            ConvBlockDownSample(64, 128, 2),
            ConvBlockDownSample(128, 192, 2),
            ConvBlockDownSample(192, 256, 2),
            ConvBlock(256, out_channels, 2, 1),
        )
    
    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)



data_transform = transforms.Compose([
    transforms.Resize(128),              # Resize smaller dimension to 256
    transforms.CenterCrop(96),          # Crop to 224x224 (standard input size)
    transforms.Pad(16, padding_mode="reflect"),
    transforms.ToTensor(),               # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])


test_dataset = MergedImageFolder("./data", None, (0.8, 1))

def collate_fn(examples : list):
    images = []
    classes = []
    for image, image_class in examples:
        images.append(image)
        classes.append(image_class)
    return torch.stack(images), torch.tensor(np.array(classes, dtype=int))

num_classes = len(test_dataset.classes)

model = Detector(3, num_classes).to(device)
model.load_state_dict(torch.load("model.pt"))

indexes = list(range(len(test_dataset)))
random.shuffle(indexes)
total_predictions = 0
correct_predictions = 0
for i in indexes:
    image, class_ = test_dataset[i]
    
    input = data_transform(image).to(device).unsqueeze(0)
    logits = model(input)
    probs = torch.softmax(logits, 1)
    B, C, W, H = probs.shape
    mid = W//2, H//2
    output = probs[:, :, *mid]
    predicted_class = torch.argmax(output.squeeze(0)).item()
    total_predictions += 1
    correct_predictions += predicted_class == class_
    print(f"predicted class: {test_dataset.classes[predicted_class]} \ttrue class: {test_dataset.classes[class_]} \taccuracy: {correct_predictions/total_predictions:.3f}")
    image = np.array(image)
    cv2.imshow("image", image)
    key = cv2.waitKey(-1) & 0xff
    if key == 27 or key == ord("q"): # esc or q
        break







