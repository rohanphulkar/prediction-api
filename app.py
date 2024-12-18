from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Backupdoc ML Service", 
    description="API for classifying dental x-ray images as OPG or regular x-ray",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Load the .tflite model
tflite_interpreter = tf.lite.Interpreter(
    model_path="opg_xray_classifier.tflite",
    num_threads=4  # Optimize TFLite threading
)
tflite_interpreter.allocate_tensors()

# Cache model details to avoid repeated lookups
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()

# Define image size constant
image_size = (150, 150)

@lru_cache(maxsize=1024)  # Cache recent predictions
def process_image(image_bytes):
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.resize(image_size, Image.LANCZOS)  # Use LANCZOS for better quality
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    data = np.array(image, dtype=np.float32)
    data = data / 255.0
    return np.expand_dims(data, axis=0)

def predict_image_class(image_data):
    """Make prediction using TFLite model"""
    try:
        if image_data.shape[-1] != 3:
            raise ValueError("Input image must have 3 channels (RGB)")
            
        tflite_interpreter.set_tensor(input_details[0]['index'], image_data)
        tflite_interpreter.invoke()
        predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        
        return "opg" if predictions[0][0] < 0.5 else "xray"
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

async def process_and_predict(image_bytes):
    """Async wrapper for CPU-intensive operations"""
    loop = asyncio.get_event_loop()
    # Process image in thread pool
    image_data = await loop.run_in_executor(
        thread_pool,
        process_image,
        image_bytes
    )
    # Make prediction in thread pool
    result = await loop.run_in_executor(
        thread_pool,
        predict_image_class,
        image_data
    )
    return result

@app.post("/predict", 
    summary="Predict image class",
    description="Upload a dental x-ray image and get prediction whether it's an OPG or regular x-ray",
    response_description="Returns prediction result as 'opg' or 'xray'",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {"prediction": "opg"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error processing image"}
                }
            }
        }
    }
)
async def predict(file: UploadFile = File(..., description="Dental x-ray image file to classify")):
    try:
        contents = await file.read()
        class_name = await process_and_predict(contents)
        return JSONResponse(
            status_code=200,
            content={"prediction": class_name}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/",
    summary="Root endpoint",
    description="Welcome message endpoint",
    response_description="Returns welcome message",
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {"message": "Welcome to the Backupdoc ML Service"}
                }
            }
        }
    }
)
async def root():
    return JSONResponse(
        status_code=200,
        content={"message": "Welcome to the Backupdoc ML Service"}
    )
