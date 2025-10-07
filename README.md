# MNIST Digit Classifier with Triton Inference Server

Complete beginner-friendly project for deploying a TensorFlow model with NVIDIA Triton.

## 📋 Prerequisites

- Docker installed
- Python 3.8+
- Basic understanding of machine learning

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
pip install tensorflow numpy pillow flask tritonclient[all] tf2onnx
```

### 2. Train and Export Model

```bash
python train_and_export.py
```

This will:
- Download MNIST dataset
- Train a CNN model (5 epochs, ~99% accuracy)
- Export to SavedModel format
- Export to ONNX format (optional)
- Create `model_repository/mnist_savedmodel/1/` directory

### 3. Create Triton Configuration

Create `model_repository/mnist_savedmodel/config.pbtxt`:

```protobuf
name: "mnist_savedmodel"
platform: "tensorflow_savedmodel"
max_batch_size: 8

input [
  {
    name: "input_1"
    data_type: TYPE_FP32
    dims: [ 28, 28, 1 ]
  }
]

output [
  {
    name: "dense_1"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]

dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

**Note**: Update `input_1` and `dense_1` names if your model uses different layer names. Check with:

```python
import tensorflow as tf
model = tf.keras.models.load_model('model_repository/mnist_savedmodel/1')
print("Input:", model.input.name)
print("Output:", model.output.name)
```

### 4. Start Triton Server

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

Wait for: `Started HTTPService at 0.0.0.0:8000`

### 5. Test with Python Client

```bash
python triton_client.py path/to/digit_image.png
```

### 6. Launch Web Interface

```bash
# Create templates directory
mkdir -p templates

# Start Flask app
python app.py
```

Visit: `http://localhost:5000`

## 📁 Project Structure

```
.
├── train_and_export.py       # Model training script
├── triton_client.py           # Command-line test client
├── app.py                     # Flask web server
├── templates/
│   └── index.html             # Web interface
└── model_repository/
    └── mnist_savedmodel/
        ├── config.pbtxt       # Triton configuration
        └── 1/                 # Model version
            ├── saved_model.pb
            └── variables/
```

## 🎯 Skills Learned

### 1. Model Export
- ✅ TensorFlow SavedModel format
- ✅ ONNX conversion (cross-framework)
- ✅ Model versioning

### 2. Triton Configuration
- ✅ Input/output tensor specifications
- ✅ Dynamic batching
- ✅ Platform selection (TensorFlow/ONNX)
