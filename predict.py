import torch
from torchvision import transforms
from PIL import Image
from models.cnn_model import SimpleCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # covid, lung opacity, normal, viral pneumonia

# Map class indices to labels
class_names = ['COVID', 'Lung Opacity', 'Normal', 'Viral_Pneumonia']

# Load model
model = SimpleCNN(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load('models/cnn_model.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_covid(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
