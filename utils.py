import numpy as np
from PIL import Image
import cv2
import tensorflow as tf


def preprocess_image(image, target_size=(224, 224), brightness_factor=1.3):
    """
    Preprocess image for VGG16 model input
    Uses VGG16-specific preprocessing
    """
    # Convert PIL image to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Resize image
    image = np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    image = cv2.resize(image, target_size)
    
    # VGG16 preprocessing 
    try:

        from tensorflow.keras.applications.vgg16 import preprocess_input
        image_batch = np.expand_dims(image, axis=0)
        processed_image = preprocess_input(image_batch)
    except ImportError:

        image = image[..., ::-1]
        # Subtract ImageNet mean
        image = image.astype(np.float32)
        image -= np.array([103.939, 116.779, 123.68])
        processed_image = np.expand_dims(image, axis=0)
    
    return processed_image

def get_class_labels():
    return [
        "arachnoid-cyst",
        "cerebellah-hypoplasia",
        "colphocephaly",
        "encephalocele",
        "mild-ventriculomegaly",
        "moderate-ventriculomegaly",
        "normal",
        "polencephaly",
        "severe-ventriculomegaly"
    ]

def format_prediction_result(prediction, class_labels=None):
    if class_labels is None:
        class_labels = get_class_labels()
    
    class_index = np.argmax(prediction[0])
    confidence = float(prediction[0][class_index])
    
    confidence_pairs = [(i, float(conf)) for i, conf in enumerate(prediction[0]) if i < len(class_labels)]
    
    confidence_pairs.sort(key=lambda x: x[1], reverse=True)
    
    potentials = confidence_pairs[1:3]
    
    top_confidences = {}
    for idx, conf in potentials:
        top_confidences[class_labels[idx]] = round(conf * 100, 2)
    
    result = {
        "predicted_class": class_labels[class_index] if class_index < len(class_labels) else str(class_index),
        "confidence": round(confidence * 100, 2), 
        "potential": top_confidences
    }
    
    return result

# Define the softmax function as a replacement for softmax_v2
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# Function to load model with custom objects to handle softmax_v2
def load_model_with_custom_objects(model_path):

    # Define custom objects to handle problematic serialization
    custom_objects = {
        'softmax_v2': tf.nn.softmax,  # Map softmax_v2 to standard softmax
        'Adam': tf.keras.optimizers.legacy.Adam,
    }
    
    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects=custom_objects
        )
        print("Model loaded successfully with custom objects and compile=False")
        return model
        
    except Exception as e:
        print(f"First loading attempt failed: {str(e)}")
        
        # Second attempt: Try with different custom objects
        try:
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'softmax': tf.nn.softmax},
                compile=False
            )
            print("Model loaded successfully with second approach")
            return model
            
        except Exception as e2:
            print(f"Second loading attempt failed: {str(e2)}")
            
            