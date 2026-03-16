import os, json, argparse
from typing import Any, cast

# Lazy imports - only import when needed to avoid numpy issues
_model = None
_device = None
_transform = None
_classes = None


def _extract_state_dict(checkpoint):
    """Support both wrapped checkpoints and raw state dict checkpoints."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Loading model from checkpoint dictionary...")
        return checkpoint["model_state_dict"]

    if isinstance(checkpoint, dict):
        print("Loading model from raw state dict...")
        return checkpoint

    raise ValueError("Unsupported checkpoint format")


def _build_model(EfficientNet, torch_nn, num_classes, sequential_fc=True):
    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    if sequential_fc:
        # Some checkpoints store FC keys as _fc.0.weight / _fc.0.bias.
        model._fc = torch_nn.Sequential(torch_nn.Linear(num_ftrs, num_classes))
    else:
        # Other checkpoints store FC keys as _fc.weight / _fc.bias.
        model._fc = torch_nn.Linear(num_ftrs, num_classes)
    return model


def _predict_probabilities(img_path):
    """Run a forward pass and return probabilities in 0-1 range."""
    # Lazy load dependencies on first call
    load_dependencies()

    if _transform is None or _device is None or _model is None:
        raise RuntimeError("Model dependencies failed to initialize")

    # Use cv2 to load image (more compatible with numpy than PIL)
    img_cv = cv2.imread(img_path)
    if img_cv is None:
        raise ValueError(f"Failed to load image: {img_path}")

    # Convert BGR to RGB (cv2 loads in BGR by default)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Convert to PIL for torchvision transforms
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray(img_rgb)

    # Apply transforms
    x = cast(Any, _transform(img_pil)).unsqueeze(0).to(_device)

    # Get prediction using sigmoid for multi-label
    with torch.no_grad():
        logits = _model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs

def load_dependencies():
    """Lazy load all ML dependencies when first needed"""
    global _model, _device, _transform, _classes, torch, torch_nn, transforms, EfficientNet, cv2, np
    
    if _model is not None:
        return  # Already loaded
    
    # Import numpy FIRST to ensure it's available
    import numpy as np
    # Make sure numpy array operations work
    np.array([1, 2, 3])  # Test numpy
    
    # Now import torch and torchvision
    import torch
    import torch.nn as torch_nn
    from torchvision import transforms
    from efficientnet_pytorch import EfficientNet
    import cv2
    
    # Config - use absolute paths relative to this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(SCRIPT_DIR, "bat_28.pth")
    CLASSES_PATH = os.path.join(SCRIPT_DIR, "classes_28.json")
    
    # Load classes
    with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
        _classes = json.load(f)
    
    # Device & transforms
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Load model
    def load_model(model_path, num_classes):
        checkpoint = torch.load(model_path, map_location=_device)
        state_dict = _extract_state_dict(checkpoint)

        # Try FC layout expected by many training checkpoints first.
        model = _build_model(EfficientNet, torch_nn, num_classes, sequential_fc=True)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # Fallback for checkpoints that store _fc.weight/_fc.bias.
            model = _build_model(EfficientNet, torch_nn, num_classes, sequential_fc=False)
            try:
                model.load_state_dict(state_dict)
            except RuntimeError:
                raise RuntimeError(
                    f"Could not load model checkpoint with either FC layout: {e}"
                )

        model.eval().to(_device)
        return model
    
    _model = load_model(MODEL_PATH, len(_classes))

def classify_image(img_path, threshold=0.01):
    """
    Classify an image using multi-label classification and return top species.
    Returns the top predicted species and its confidence.
    
    Args:
        img_path: Path to spectrogram image
        threshold: Minimum confidence threshold (0-1)
    
    Returns:
        (species_name, confidence_percentage)
    """
    probs = _predict_probabilities(img_path)
    classes = _classes
    if classes is None:
        raise RuntimeError("Class labels failed to initialize")
    
    # Get all detections above threshold
    detections = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            if i >= len(classes):
                continue
            detections.append({
                'species': classes[i],
                'confidence': float(prob * 100)
            })
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # Return top prediction
    if detections:
        top = detections[0]
        species = top['species'].replace(" ", "_")
        return species, round(top['confidence'], 2)
    
    return "Unknown_species", 0.0

def classify_image_multi(img_path, threshold=0.01):
    """
    Classify an image and return multiple species predictions.
    
    Args:
        img_path: Path to spectrogram image
        threshold: Minimum confidence threshold (0-1)
    
    Returns:
        List of (species_name, confidence_percentage) tuples
    """
    probs = _predict_probabilities(img_path)
    classes = _classes
    if classes is None:
        raise RuntimeError("Class labels failed to initialize")
    
    detections = []
    for i, prob in enumerate(probs):
        if prob >= threshold:
            if i >= len(classes):
                continue
            detections.append((classes[i].replace(" ", "_"), float(prob * 100)))
    
    # Sort by confidence (descending)
    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    
    return detections

# --- CLI Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify a bat spectrogram image using EfficientNet.")
    parser.add_argument('image_path', type=str, help='Path to the spectrogram image file (e.g., spectrogram.jpg)')
    
    args = parser.parse_args()
    
    prediction, confidence = classify_image(args.image_path)
    
    print("\n--- Final Result ---")
    print(f"Predicted Species: {prediction}")
    print(f"Confidence: {confidence}%")
    