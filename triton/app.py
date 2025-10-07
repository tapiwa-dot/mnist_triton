from flask import Flask, render_template, request, jsonify
import tritonclient.http as httpclient
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# TRITON CONFIG
TRITON_URL = "localhost:8000"
MODEL_NAME = "mnist_savedmodel"

def preprocess_image(image_bytes):
    """Preprocess uploaded image for MNIST model"""
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

def get_prediction(image_data):
    """Send inference request to Triton server"""
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)

        # Prepare input
        inputs = []
        infer_input = httpclient.InferInput('input_1', image_data.shape, "FP32")
        infer_input.set_data_from_numpy(image_data)
        inputs.append(infer_input)

        # Prepare output (update 'dense_2' if your model output name differs)
        outputs = [httpclient.InferRequestedOutput('dense_1')]

        # Send request
        response = triton_client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )

        # Extract results
        logits = response.as_numpy('dense_1')
        predicted_class = int(np.argmax(logits))
        probabilities = logits[0].tolist()

        return {
            'predicted_digit': predicted_class,
            'confidence': float(max(probabilities) * 100),
            'all_probabilities': [float(p * 100) for p in probabilities]
        }

    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected!'}), 400

    try:
        # Read and preprocess image
        image_bytes = file.read()
        image_data = preprocess_image(image_bytes)

        # Get prediction
        result = get_prediction(image_data)

        # Encode image for frontend
        img_base64 = base64.b64encode(image_bytes).decode('utf-8')
        result['image'] = f"data:image/png;base64,{img_base64}"

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    try:
        triton_client = httpclient.InferenceServerClient(url=TRITON_URL)
        is_live = triton_client.is_server_live()
        is_ready = triton_client.is_model_ready(MODEL_NAME)
        return jsonify({
            'triton_live': is_live,
            'model_ready': is_ready
        })
    except Exception as e:
        return jsonify({
            'triton_live': False,
            'model_ready': False,
            'error': str(e)
        }), 503

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
