import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def get_data_loader(data_dir, batch_size=32, shuffle=True):
    dataset = datasets.ImageFolder(root=data_dir, transform=image_transforms)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0,      # Set to 0 to prevent CPU multi-process hanging
        pin_memory=False    # Prevents memory locking issues on CPU
    )
    return dataset, loader
    
if __name__ == "__main__":
    # Path to the extracted dataset directory
    DATA_DIR = "./dataset/PlantVillage"
    
    # Initialize the dataset and dataloader
    dataset, loader = get_data_loader(DATA_DIR, batch_size=32, shuffle=True)
    
    # Print basic summary
    print(f"Total images loaded: {len(dataset)}")
    print(f"Total classes found: {len(dataset.classes)}")
    print(f"Detected classes: {dataset.classes[:3]}...") # Displays first 3 disease categories
    
    # Retrieve a single batch to verify structure
    images, labels = next(iter(loader))
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch label tensor shape: {labels.shape}")