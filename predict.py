import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
from model import build_model

def predict_leaf_disease(image_path, model_path="plant_disease_model.pth"):
    # 1. Define same input transforms used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. Load and sanitize input image
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[SECURITY/ERROR] Failed to read image file: {e}")
        return

    # 3. Transform image and add Batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
    input_tensor = transform(raw_image).unsqueeze(0)

    # 4. Load trained model weights
    model = build_model(num_classes=15)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()  # Set to evaluation mode (disables dropout)

    # 5. Run forward pass with gradient tracking turned off
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    print(f"\n--- Prediction Results ---")
    print(f"Target Image: {image_path}")
    print(f"Predicted Class Index: {predicted_class.item()}")
    print(f"Confidence Score: {confidence.item() * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        predict_leaf_disease(img_path)
    else:
        print("Usage: python predict.py <path_to_leaf_image>")