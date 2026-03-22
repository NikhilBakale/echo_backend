import os, json, argparse
from typing import Any, cast

# Lazy imports - only import when needed to avoid numpy issues
_model = None
_device = None
_transform = None
_classes = None


def _is_supported_efficientnet_state_dict(state_dict):
    """Return True for checkpoints compatible with this inference model."""
    keys = list(state_dict.keys())
    if not keys:
        return False

    # This predictor expects plain EfficientNet keys like `_conv_stem.*`.
    # Custom checkpoints with `backbone.*` need a different architecture.
    has_plain_backbone = any(k.startswith("_conv_stem") for k in keys)
    has_wrapped_backbone = any(k.startswith("backbone.") for k in keys)
    return has_plain_backbone and not has_wrapped_backbone


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

    # Match reference predictor image-loading path (PIL -> RGB).
    from PIL import Image as PILImage
    img_pil = PILImage.open(img_path).convert('RGB')

    # Apply transforms
    x = cast(Any, _transform(img_pil)).unsqueeze(0).to(_device)

    # Get prediction using sigmoid for multi-label
    with torch.no_grad():
        logits = _model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    return probs

def load_dependencies():
    """Lazy load all ML dependencies when first needed"""
    global _model, _device, _transform, _classes, torch, torch_nn, transforms, EfficientNet, np
    
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
    # Config - use absolute paths relative to this script's location
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    candidate_pairs = [
        # Preferred: known plain EfficientNet checkpoint with matching class file.
        ("bat_28.pth", "classes_28.json"),
        # Secondary: older plain EfficientNet checkpoint.
        ("efficientnet_b0_bat_3_dataset(1).pth", "new_3_dataset_classes(1).json"),
        # Last: legacy default in this file; may be incompatible with this architecture.
        ("bat_optimized_final.pth", "newclass.json"),
    ]
    
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

        if not isinstance(state_dict, dict) or not _is_supported_efficientnet_state_dict(state_dict):
            raise RuntimeError(
                f"Checkpoint is not a plain EfficientNet state_dict compatible with this predictor: {model_path}"
            )

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

    load_errors = []
    for model_name, classes_name in candidate_pairs:
        model_path = os.path.join(SCRIPT_DIR, model_name)
        classes_path = os.path.join(SCRIPT_DIR, classes_name)

        if not (os.path.exists(model_path) and os.path.exists(classes_path)):
            load_errors.append(f"missing files: {model_name} / {classes_name}")
            continue

        try:
            with open(classes_path, 'r', encoding='utf-8') as f:
                classes = json.load(f)

            model = load_model(model_path, len(classes))
            _model = model
            _classes = classes
            print(f"Loaded model checkpoint: {model_name}")
            print(f"Loaded classes file: {classes_name} ({len(classes)} classes)")
            break
        except Exception as e:
            load_errors.append(f"{model_name}: {e}")

    if _model is None or _classes is None:
        details = "\n".join(load_errors) if load_errors else "No compatible checkpoint candidates found."
        raise RuntimeError(f"Failed to load any compatible model checkpoint.\n{details}")

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
    parser.add_argument(
        '--multi',
        action='store_true',
        help='Return all species above threshold instead of only the top prediction.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help='Minimum confidence threshold in 0-1 range (default: 0.01).',
    )
    
    args = parser.parse_args()

    if args.multi:
        detections = classify_image_multi(args.image_path, threshold=args.threshold)
        print("\n--- Multi-Species Results ---")
        print(f"Threshold: {args.threshold}")
        if detections:
            for species, confidence in detections:
                print(f"- {species}: {confidence:.2f}%")
        else:
            print("No species detected above threshold.")
    else:
        prediction, confidence = classify_image(args.image_path, threshold=args.threshold)
        print("\n--- Final Result ---")
        print(f"Predicted Species: {prediction}")
        print(f"Confidence: {confidence}%")
    