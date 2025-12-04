import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from dataloader import MergedImageFolder
import numpy as np
from torchvision import transforms

batch_size = 32
epochs = 100
lr = 2e-3
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
            ConvBlock(256, 256, 2, 1),
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


train_dataset = MergedImageFolder("./data", data_transform, 0.8)
test_dataset = MergedImageFolder("./data", data_transform, (0.8, 1))

def collate_fn(examples : list):
    images = []
    classes = []
    for image, image_class in examples:
        images.append(image)
        classes.append(image_class)
    return torch.stack(images), torch.tensor(np.array(classes, dtype=int))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
num_classes = len(train_dataset.classes)

model = Detector(3, num_classes).to(device)
classification_head = nn.Linear(256, num_classes)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr)


for epoch in range(epochs):
    train_loss = 0
    for images, classes in train_dataloader:
        images = images.to(device, non_blocking=True)
        classes = classes.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, 1)
        B, C, W, H = probs.shape
        mid = W//2, H//2
        #print(W, H)
        output = probs[:, :, *mid]
        loss = loss_fn(output, classes)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for images, classes in test_dataloader:
            images = images.to(device, non_blocking=True)
            classes = classes.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, 1)
            B, C, W, H = probs.shape
            mid = W//2, H//2
            output = probs[:, :, *mid]
            loss = loss_fn(output, classes)
            test_loss += loss.item()
            correct_predictions += (output.argmax(1) == classes).float().sum()
    test_loss /= len(test_dataloader)
    train_loss /= len(train_dataloader)
    correct_predictions /= len(test_dataloader)
    print(f"epoch: {epoch+1}/{epochs} \ttest loss: {test_loss:.3f} \ttrain loss: {train_loss:.3f} \taccuracy: {correct_predictions:.3f}")

torch.save(model.state_dict(), "model.pt")


