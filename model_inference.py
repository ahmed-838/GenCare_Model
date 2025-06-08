from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="XJZUmFkSMQsTwjy6fAPF"
)

# List of conditions we want to detect
TARGET_CONDITIONS = [
    "moderate-ventriculomegaly",
    "cerebellah-hypoplasia",
    "normal",
    "polencephaly",
    "encephalocele",
    "mild-ventriculomegaly",
    "severe-ventriculomegaly",
    "arachnoid-cyst",
    "colphocephaly"
]

def get_prediction(image_path):
    """
    Get predictions for an image using the Roboflow model
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: The prediction results
    """
    try:
        # Get original result from model
        result = CLIENT.infer(image_path, model_id="fetal-brain-abnormalities-ultrasound/1")
        
        # Filter the predictions to only include target conditions
        filtered_predictions = {}
        for condition in TARGET_CONDITIONS:
            if condition in result.get("predictions", {}):
                filtered_predictions[condition] = result["predictions"][condition]
        
        # Update the result dictionary with filtered predictions
        result["predictions"] = filtered_predictions
        
        # Get the predicted class (highest confidence)
        predicted_classes = result.get("predicted_classes", [])
        
        # Check if "normal" is the predicted class
        if "normal" in predicted_classes:
            result["diagnosis_message"] = "no abnormalities detected"
        else:
            # Filter predicted_classes to only include our target conditions
            filtered_classes = [cls for cls in predicted_classes if cls in TARGET_CONDITIONS]
            result["predicted_classes"] = filtered_classes
            
            if filtered_classes:
                result["diagnosis_message"] = f"detected: {', '.join(filtered_classes)}"
            else:
                result["diagnosis_message"] = "no abnormalities detected"
                
        return result
    except Exception as e:
        print(f"Error during inference: {e}")
        raise e