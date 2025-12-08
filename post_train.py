import os
import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from dataloader import MergedImageFolder
import numpy as np
from torchvision import transforms
import time
import datetime
from torch.utils.tensorboard import SummaryWriter



batch_size = 32
epochs = 50 # training limit. if early stopping hasnt been reached
lr = 1e-4
head_lr = 1e-2
betas = (0.9, 0.999)
early_stopping_patience = 50
device = "cuda"
load_checkpoint = True
load_path = "model.pt"
save_path = "model_post_train.pt"
log_dir = ("logs_with_extra/" if load_checkpoint else "logs/") + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = SummaryWriter(log_dir)


class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, mid_ratio : float, kernel_size : int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_ratio = mid_ratio
        self.mid_channels = int((in_channels + out_channels) * mid_ratio / 2)
        assert kernel_size % 2 == 1, f"kernel size must be odd. got {kernel_size}"
        self.up_conv = nn.Conv2d(in_channels, self.mid_channels, kernel_size, 1, (kernel_size-1)//2, padding_mode="reflect")
        self.down_conv = nn.Conv2d(self.mid_channels, out_channels, 1, 1, 0, padding_mode="reflect")
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, padding_mode="reflect")
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
        self.up_conv = nn.Conv2d(in_channels, self.mid_channels, kernel_size, 1, (kernel_size-1)//2, padding_mode="reflect")
        self.down_conv = nn.Conv2d(self.mid_channels, out_channels, 4, 2, 1, padding_mode="reflect") # down samples by 2x
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 2, 2, 0, padding_mode="reflect") # also down samples by 2x, but uses 2x2 kernel
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
            ConvBlock(32, 48, 2),
            ConvBlockDownSample(48, 64, 2),
            ConvBlock(64, 96, 2),
            ConvBlockDownSample(96, 128, 2),
            ConvBlockDownSample(128, 192, 2),
            ConvBlockDownSample(192, 256, 2),
            ConvBlock(256, out_channels, 2, 1),
        )
    
    def forward(self, x : Tensor) -> Tensor:
        return self.layers(x)



train_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.RandomCrop(96),
    transforms.Pad(32, padding_mode="reflect"),
    transforms.ToTensor(),               # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

test_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(96),
    transforms.Pad(32, padding_mode="reflect"),
    transforms.ToTensor(),               # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])


train_datasets = [
                MergedImageFolder("my_data/data/", train_transform, use_fraction=0.5, dataset_nr=2),
                ]
test_datasets = [
                MergedImageFolder("my_data/data/", test_transform, use_fraction=(0.5,1), dataset_nr=2),
                ]

def collate_fn(examples : list):
    images = []
    classes = []
    for image, image_class in examples:
        images.append(image)
        classes.append(image_class)
    return torch.stack(images), torch.tensor(np.array(classes, dtype=int))

printed = []
def printonce(*args):
    text = " ".join(map(str, args))
    if text not in printed:
        print(text)
        printed.append(text)


total_test_size = sum(map(len, test_datasets))
# calculate batch size for each dataset in order to have equal total batch = batch_size
train_dataset_sizes = list(map(len, train_datasets))
total_train_size = sum(train_dataset_sizes)
num_batches = -(-total_train_size//batch_size)
batch_sizes = [round(len(train_dataset)/num_batches) for train_dataset in train_datasets]
print(f"dataset sizes: {train_dataset_sizes}")
print(f"batch sizes: {batch_sizes}")
print(f"total batch size: {sum(batch_sizes)}")
print(f"batch counts: {[-(-dataset_size//batch_size) for batch_size, dataset_size in zip(batch_sizes, train_dataset_sizes)]}")
train_dataloaders = [torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn) for train_dataset, batch_size in zip(train_datasets, batch_sizes)]
test_dataloaders = [torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn) for test_dataset in test_datasets]



model = Detector(3, 256).to(device).bfloat16()
classification_heads = [ConvBlock(256, len(train_dataset.classes), 1, 1).to(device) for train_dataset in train_datasets]

print()
print(f"model numel: {sum([param.numel() for param in model.parameters()])}")


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr, betas=betas)
head_optimizers = [optim.AdamW(head.parameters(), head_lr, betas=betas) for head in classification_heads]



if load_checkpoint and os.path.exists(load_path):
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict["backbone"])
    optimizer.load_state_dict(state_dict["backbone_optim"])
    print("loaded checkpoint")




accuracy_history = []
start_time = time.time()
try:
    for epoch in range(epochs):
        train_loss = 0
        total_examples = 0
        for dataset_batches in zip(*train_dataloaders):
            for (images, classes), head, head_optim in zip(dataset_batches, classification_heads, head_optimizers):
                images = images.to(device, torch.bfloat16, non_blocking=True)
                classes = classes.to(device, non_blocking=True)
                #with torch.amp.autocast(device, torch.bfloat16):
                features = model(images).float()
                logits = head(features)
                probs = torch.softmax(logits, 1)
                B, C, W, H = probs.shape
                printonce(W, H)
                mid = W//2, H//2
                output = probs[:, :, *mid]
                loss = loss_fn(output, classes)
                train_loss += loss.item()*B
                total_examples += B
                loss.backward()
                head_optim.step()
                head_optim.zero_grad()
            optimizer.step()
            optimizer.zero_grad()
        test_loss = 0
        correct_predictions = [0 for i in range(len(test_datasets))]
        total_predictions = [0 for i in range(len(test_datasets))]
        with torch.no_grad():
            for i, (test_dataloader, head) in enumerate(zip(test_dataloaders, classification_heads)):
                for images, classes in test_dataloader:
                    images = images.to(device, torch.bfloat16, non_blocking=True)
                    classes = classes.to(device, non_blocking=True)
                    #with torch.amp.autocast(device, torch.bfloat16):
                    features = model(images).float()
                    logits = head(features)
                    probs = torch.softmax(logits, 1)
                    B, C, W, H = probs.shape
                    mid = W//2, H//2
                    output = probs[:, :, *mid]
                    loss = loss_fn(output, classes)
                    test_loss += loss.item() * B
                    correct_predictions[i] += (output.argmax(1) == classes).float().sum().item()
                    total_predictions[i] += B
        test_loss /= sum(total_predictions)
        train_loss /= total_examples
        correct_predictions = [correct/total for correct, total in zip(correct_predictions, total_predictions)]
        for i, acc in enumerate(correct_predictions):
            log_writer.add_scalar(f"accuracy/data{i}", acc, global_step=epoch)
        accuracy = np.mean(correct_predictions)
        accuracy_history.append(accuracy)
        # early stopping
        print(f"epoch: {epoch+1}/{epochs} \ttest loss: {test_loss:.3f} \ttrain loss: {train_loss:.3f}" \
                f" \taccuracy: ({', '.join([f'{correct_prediction:.3f}' for correct_prediction in correct_predictions])}){'\'' if accuracy_history.index(max(accuracy_history)) == len(accuracy_history)-1 else ' '}" \
                f"\tmean epoch time: {(time.time() - start_time)/(epoch+1):.2f}")
        if accuracy_history.index(max(accuracy_history)) < len(accuracy_history) - early_stopping_patience:
            print("early stopping")
            break # stop training if accuray hasn't increased in the last 10 epochs
except KeyboardInterrupt:
    save_model = input("save model(Y/n): ").lower() != "n"
    if save_model:
        pass
    else:
        exit()
except:
    save_path = "model_backup.pt"

state_dict = {
    "backbone" : model.state_dict(),
    "backbone_optim" : optimizer.state_dict(),
    "heads" : [head.state_dict() for head in classification_heads],
    "head_optims" : [optim.state_dict() for optim in head_optimizers],
}
torch.save(state_dict, save_path)
print("model saved")

