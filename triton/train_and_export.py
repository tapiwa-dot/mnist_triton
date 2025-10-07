import tensorflow as tf 
from tensorflow import keras
from keras import layers
import numpy as np
import tf2onnx

# Define the CNN model
def create_mnist_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    return model

def train_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create and compile model
    model = create_mnist_model()
    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    model.summary()
    
    # Train the model 
    print("\n Training model...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on test set
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n Test accuracy: {test_acc:.4f}")
    
    return model

def export_to_savedmodel(model, export_path="./model_repository/mnist_savedmodel/1/model.savedmodel/"):
    """Export model in Tensorflow savedmodel format (native triton)"""
    print(f" \n Exporting to saved model format at {export_path}")
    model.save(export_path)
    print("Model export successfully!!!!!")
    
def export_to_onnx(model, export_path="mnist.onnx"):
    """Export model to ONNX format (alternative)"""
    print(f"\n Exporting to ONNX format at {export_path}")
    
    #Get model signature 
    spec = (tf.TensorSpec((None, 28, 28, 1), tf.float32, name="input"),)
    
    # Convert to ONNX 
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=export_path
    )
    
    print("ONNX model export successfully!!!!")
    
if __name__=="__main__":
    # Train the model 
    model = train_model()
    
    # Export in the both formats 
    # Option 1: Savedmodel (recommended for Tensorflow models)
    export_to_savedmodel(model)
    
    # Option 2: ONNX(cross-framework format)
    export_to_onnx(model)
    
    print("\n Training and export complete")
    print("Next steps")
    print("1. Create Triton config.pbtxt")
    print("2. Start Triton server")
    print("3. Send inference requests")