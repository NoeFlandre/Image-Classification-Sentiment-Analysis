import torch
import torchvision.transforms as transforms
from PIL import Image
import json

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)  # Add batch dimension
    return batch_t

def get_prediction(image_tensor, model):
    # Inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get the predicted class (index with highest probability)
    _, predicted_class = torch.max(output, 1)
    
    # Load ImageNet class labels
    with open('imagenet_class_index.json') as f:
        labels = json.load(f)
    
    class_id = predicted_class.item()
    class_name = labels[str(class_id)][1]  # Return class name
    
    return class_id, class_name
