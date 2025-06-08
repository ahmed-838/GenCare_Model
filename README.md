# Image Analysis API

This Flask API allows you to upload images and get analysis results from a pre-trained model for fetal brain abnormalities ultrasound. The API is configured to detect 9 specific conditions.

## Target Conditions

The API is configured to detect only the following conditions:
1. moderate-ventriculomegaly
2. cerebellah-hypoplasia
3. normal
4. polencephaly
5. encephalocele
6. mild-ventriculomegaly
7. severe-ventriculomegaly
8. arachnoid-cyst
9. colphocephaly

## Project Structure

```
model-inference-app/
├── app.py                # Main Flask API application
├── model_inference.py    # Model client and inference handling
├── requirements.txt      # Python dependencies
├── uploads/              # Temporary storage for uploaded files
└── README.md             # Documentation
```

## Setup Instructions

1. **Create and activate virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API server**:
   ```bash
   python app.py
   ```

   The API will be available at `http://127.0.0.1:5000`.

## API Endpoints

### Health Check
- **URL:** `/health`
- **Method:** `GET`
- **Response:** `{"status": "healthy"}`

### Get Target Conditions
- **URL:** `/api/conditions`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "success": true,
    "target_conditions": ["moderate-ventriculomegaly", "cerebellah-hypoplasia", "normal", ...]
  }
  ```

### Prediction
- **URL:** `/api/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Request Body:**
  - `file`: Image file (jpg, jpeg, png)
- **Response:**
  ```json
  {
    "success": true,
    "filename": "example.jpg",
    "diagnosis_message": "تم اكتشاف: moderate-ventriculomegaly",
    "results": {
      "predicted_classes": ["moderate-ventriculomegaly"],
      "predictions": {
        "moderate-ventriculomegaly": {
          "class_id": 11,
          "confidence": 0.7707487940788269
        },
        // Other filtered predictions
      }
      // Other result data
    }
  }
  ```

## Testing with Postman

1. **Open Postman** and create a new request
2. Set the request type to `POST`
3. Enter the URL: `http://127.0.0.1:5000/api/predict`
4. Go to the "Body" tab
5. Select "form-data"
6. Add a key named "file" and change the type from "Text" to "File"
7. Click "Select Files" and choose an image file
8. Click "Send" to make the request
9. View the JSON response with the analysis results

## Error Handling

The API returns appropriate HTTP status codes:
- `400`: Bad Request (missing file, invalid file type)
- `500`: Internal Server Error (model inference issues)

Each error response includes:
```json
{
  "error": "Error description",
  "success": false
}
```