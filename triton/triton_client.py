import tritonclient.http as httpclient
import numpy as np
from PIL import Image 
import sys

def preprocess_image(image_path):
    """Load and preprocess image MNIST model"""
    img = Image.open(image_path).convert('L')
    
    #Resize to 28x28
    img = img.resize((28, 28))
    
    #Convert to numpy array and normalize 
    img_array = np.array(img).astype('float32') / 255.0
    
    # Reshape to (1, 28, 28, 1) for batchsize 1
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array

def infer_digit(image_path, url='localhost:8000', model_name='mnist_savedmodel'):
    """Send inference request to Triton server"""
    
    # Create Triton Client 
    triton_client = httpclient.InferenceServerClient(url=url)
    
    # Check server health
    if not triton_client.is_server_live():
        print("Error: model {model_name} is not ready")
        return None
    
    # Preprocess image 
    input_data = preprocess_image(image_path)
    
    #Prepare input 
    inputs = []
    inputs.append(httpclient.InferInput('input_1', input_data.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_data)
    
    # Prepare output request
    outputs = []
    outputs.append(httpclient.InferRequestedOutput('dense_1'))
    
    # Send Inference request
    response = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )
    
    # Get predictions
    logits = response.as_numpy('dense_1')
    predicted_class = np.argmax(logits)
    confidence = np.max(logits) * 100
    
    return predicted_class, confidence, logits[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage python triton_client.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    
    print(f"Sending inference request for : {image_path}")
    result = infer_digit(image_path)
    
    if result:
        predicted_class, confidence, all_probs = result
        print(f"\n {'='*50}")
        print(f"Predicted Digit: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"{'='*50}")
        print("\n All class probabilities: ")
        for i, prob in enumerate(all_probs):
            print(f"Digit {i}: {prob*100:.2f}%")