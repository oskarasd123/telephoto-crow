import torch
from torchvision import datasets, transforms
import os
import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence
from PIL import Image
import warnings

warnings.filterwarnings(
    "ignore", 
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images", 
    category=UserWarning,
    module='PIL.Image'
)




# 1. Define image transformations (resize, convert to tensor, normalize)
data_transform = transforms.Compose([
    transforms.Resize(256),              # Resize smaller dimension to 256
    transforms.CenterCrop(224),          # Crop to 224x224 (standard input size)
    transforms.Pad(50, padding_mode="reflect"),
    transforms.ToTensor(),               # Convert to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# 2. Define the root folder (where 9/ and 10/ are located)
# Assuming 'data_root' points to the directory containing '9' and '10'
data_root = './data/'

# 3. Create a custom dataset that iterates through your top-level folders (9 and 10)
class MergedImageFolder(datasets.DatasetFolder):
    def __init__(self, root, transform=None, use_fraction : float | tuple[float, float] = 1):
        self.data_pairs : list[tuple[str, str]] = [] # tuples of (path, class)
        self.transform = transform
        classes = set()
        # Look for all numbered directories (like 9 and 10)
        for sub_dir in os.listdir(root):
            sub_path = os.path.join(root, sub_dir, 'raw') # Go into the 'raw' folder
            if os.path.isdir(sub_path):
                image_dirs = [obj for obj in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, obj))]
                for dir in image_dirs:
                    base_dir = os.path.join(sub_path, dir)
                    image_class = dir
                    classes.add(image_class)
                    for image_name in os.listdir(base_dir):
                        image_path = os.path.join(base_dir, image_name)
                        if not os.path.isdir(image_path):
                            if image_path.endswith((".jpg", ".png", ".jpeg")):
                                self.data_pairs.append((image_path, image_class))
                            else:
                                pass #print(f"unused file: {image_path}")
                        else:
                            dir_path = image_path
                            for image_name in os.listdir(dir_path):
                                image_path = os.path.join(dir_path, image_name)
                                if not os.path.isdir(image_path):
                                    if image_path.endswith((".jpg", ".png", ".jpeg")):
                                        self.data_pairs.append((image_path, image_class))
                                    else:
                                        pass #print(f"unused file: {image_path}")
        if isinstance(use_fraction, (float, int)):
            use_fraction = (0, use_fraction)
        values = np.arange(len(self.data_pairs))/len(self.data_pairs)
        np.random.shuffle(values)
        rs = RandomState(MT19937(SeedSequence(123456789)))
        self.data_pairs = np.array(self.data_pairs)
        self.data_pairs = self.data_pairs[(values >= use_fraction[0]) & (values < use_fraction[1])]
        self.classes = list(classes)
        self.classes.sort()
        self.classes.insert(0, "")
        self.class_to_idx = {image_class : i for i, image_class in enumerate(self.classes)}
        #self.class_to_idx[""] = 0 # no class


    

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # returns the image and the class index
        path, image_class = self.data_pairs[idx]
        image = Image.open(path).convert("RGB")#.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[image_class]

if __name__ == "__main__":
    # 4. Initialize the combined dataset
    bird_dataset = MergedImageFolder(root=data_root, transform=data_transform, use_fraction=(0.1, 0.8))

    
    print(f"Total images loaded: {len(bird_dataset)}")
    print(f"Number of species detected: {len(bird_dataset.classes)}")
    print(f"First 5 species: {bird_dataset.classes[:5]}")

    # 5. Create a DataLoader for batch processing
    data_loader = torch.utils.data.DataLoader(bird_dataset, batch_size=32, shuffle=True, num_workers=4)

    # try load all items
    for batch in data_loader:
        pass

