import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

MODEL_PATH = 'best_model.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['AI', 'Real'] # Must match training order. Usually alphabetical.

def load_inference_model() -> tf.keras.Model:
    return tf.keras.models.load_model(MODEL_PATH)

def predict_image(model: tf.keras.Model, img_path: str) -> tuple[str, float]:
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Model includes preprocessing/scaling if needed, or expected inputs.
    # In train.py we passed raw inputs to EfficientNet which handles it or we should be careful.
    # EfficientNetB0 in Keras expects [0, 255] inputs by default? No, wait.
    # Keras Applications models have specific preprocessing.
    # When using `image_dataset_from_directory`, it yields float32 tensors. 
    # If we didn't add a Rescaling layer, we rely on the specific model.
    # EfficientNet: "The models expect their inputs to be float tensors of pixels with values in the [0, 255] range."
    # So `img_array` (which is 0-255) is fine.
    
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    # Sigmoid: < 0.5 is class 0, > 0.5 is class 1
    # Check alphabet order: AI (0), Real (1)
    
    if score >= 0.5:
        label = CLASS_NAMES[1]
        confidence = float(score)
    else:
        label = CLASS_NAMES[0]
        confidence = float(1.0 - score)
        
    return label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict AI vs Real image")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()
    
    try:
        model = load_inference_model()
        label, conf = predict_image(model, args.image_path)
        print(f"Prediction: {label} ({conf:.2%})")
    except Exception as e:
        print(f"Error: {e}")
