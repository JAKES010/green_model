import torch
import torch.nn as nn
import torch.optim as optim
from dataset_pipeline import get_data_loader
from model import build_model

def train():
    # 1. Hyperparameters & Settings
    DATA_DIR = "./dataset/PlantVillage"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 1  # Starting with 1 epoch to verify execution pipeline
    
    # 2. Initialize Data, Model, Loss Function, and Optimizer
    print("Loading dataset...")
    dataset, train_loader = get_data_loader(DATA_DIR, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.classes)
    
    print("Initializing MobileNetV2...")
    model = build_model(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    # Optimize only the parameters that require gradients (the new classifier layer)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    
    # 3. Execution Training Loop
    print("\n--- Starting Training ---")
    model.train()
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Clear residual gradients from prior iteration
        optimizer.zero_grad()
        
        # Forward Pass: Predict outputs from image batch
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward Pass: Calculate loss gradients
        loss.backward()
        
        # Optimizer Step: Update weights of the active classifier layer
        optimizer.step()
        
        # Track training progress statistics
        running_loss += loss.item() * images.size(0)
        _, predicted_classes = torch.max(outputs, 1)
        correct_predictions += (predicted_classes == labels).sum().item()
        total_samples += labels.size(0)
        
        # Print update every 50 mini-batches
        if (batch_idx + 1) % 50 == 0:
            current_loss = running_loss / total_samples
            current_acc = (correct_predictions / total_samples) * 100
            print(f"Batch [{batch_idx + 1}/{len(train_loader)}] | Loss: {current_loss:.4f} | Accuracy: {current_acc:.2f}%")
            
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    print(f"\nTraining Complete | Final Loss: {epoch_loss:.4f} | Final Accuracy: {epoch_acc:.2f}%")
    
    # Save the trained model parameters
    torch.save(model.state_dict(), "plant_disease_model.pth")
    print("Model weights successfully exported to 'plant_disease_model.pth'")

if __name__ == "__main__":
    train()