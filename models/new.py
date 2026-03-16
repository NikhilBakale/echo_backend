import os, json, torch
from PIL import Image
import torch.nn as nn 
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import argparse # Import argparse for CLI functionality

# Config
PROJECT_ROOT = "models"
MODEL_PATH = os.path.join(PROJECT_ROOT, "efficientnet_b0_bat_3_dataset(1).pth")
CLASSES_PATH = os.path.join(PROJECT_ROOT, "new_3_dataset_classes(1).json")
# Confidence threshold - predictions below this will be marked as unknown
# NOTE: Your original logic was slightly confusing. If confidence is below 
# the threshold, it should typically return 'Unknown species'.
# I've left the original logic (return final_prediction) but added a comment.
CONFIDENCE_THRESHOLD = 75.0
# Device & transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# Load classes
try:
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        classes = json.load(f)
except FileNotFoundError:
    print(f"Error: Classes file not found at {CLASSES_PATH}")
    # Exit or use a dummy class list for demonstration if necessary
    classes = ["Placeholder_Class_0", "Placeholder_Class_1"] # Use a dummy list if you can't exit
    # exit(1) # Uncomment to properly exit on error

# Load model
def load_model(model_path, num_classes):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except:
        # Fallback for older torch versions or different save types
        model.load_state_dict(torch.load(model_path, map_location=device)) # Removed weights_only=False as it's the default or often not needed
    model.eval().to(device)
    return model

# Preload model
try:
    model = load_model(MODEL_PATH, len(classes))
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    # exit(1) # Uncomment to properly exit on error
    model = None # Set model to None to prevent errors in classify_image if model loading fails

def classify_image(img_path):
    """
    Classify an image and return the predicted class and confidence.
    If confidence is below the threshold, return 'Unknown species' (as per standard practice,
    though your original logic returned the low-confidence prediction).
    """
    if model is None:
        return "Model Load Failed", 0.0

    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        return "Image Not Found", 0.0
    
    x = transform(img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0]
        confidence, idx = probs.max(0)
        confidence_percent = round(confidence.item() * 100, 2)
        predicted_class = classes[idx.item()]
    
    # Check confidence threshold
    if confidence_percent < CONFIDENCE_THRESHOLD:
        # NOTE: If you want to return 'Unknown species' for low confidence, uncomment the line below:
        # final_prediction = 'Unknown species' 
        
        # Keeping your original logic: return the predicted class even if confidence is low
        final_prediction = predicted_class
        print(f"Low confidence prediction: {predicted_class} ({confidence_percent}%) -> Threshold: {CONFIDENCE_THRESHOLD}%")
    else:
        final_prediction = predicted_class
        print(f"High confidence prediction: {predicted_class} ({confidence_percent}%) -> Threshold: {CONFIDENCE_THRESHOLD}%")
    
    return final_prediction, confidence_percent

# --- CLI Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a bat spectrogram image using EfficientNet.")
    parser.add_argument('image_path', type=str, help='Path to the spectrogram image file (e.g., spectrogram.jpg)')
    
    args = parser.parse_args()
    
    prediction, confidence = classify_image(args.image_path)
    
    print("\n--- Final Result ---")
    print(f"Predicted Species: {prediction}")
    print(f"Confidence: {confidence}%")