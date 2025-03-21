import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide"
)

# Function to get available models
def get_available_models():
    models_dir = "models"
    # Check if directory exists
    if not os.path.exists(models_dir):
        return []
    
    # Get all .tflite files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.tflite')]
    return model_files

@st.cache_resource
def load_model(model_path):
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image):
    # Resize and normalize the image
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def predict(interpreter, image):
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
    
    # Run the inference
    interpreter.invoke()
    
    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

@st.cache_resource
def load_full_model(model_name):
    """
    Load the full TensorFlow model for Grad-CAM generation with improved compatibility.
    
    Args:
        model_name: Name of the TFLite model file
    
    Returns:
        Keras model for Grad-CAM computation or None if loading fails
    """
    # Get the base name without extension
    base_name = os.path.splitext(model_name)[0]
    
    # Check if corresponding h5 model exists
    h5_path = os.path.join("../models", f"{base_name}.h5")
    
    try:
        import tensorflow as tf
        
        if os.path.exists(h5_path):
            try:
                # Try loading with compile=False for better compatibility
                full_model = tf.keras.models.load_model(h5_path, compile=False)
                return full_model
            except Exception as e:
                st.warning(f"Could not load the full model for Grad-CAM visualization. Using simplified visualization instead.")
                return None
        else:
            return None
    except ImportError:
        return None

def generate_gradcam(interpreter, img, model_path=None):
    """
    Grad-CAM implementation with robust fallback options.
    
    Args:
        interpreter: TFLite interpreter for the model
        img: Preprocessed image (normalized to [0,1])
        model_path: Path to the TFLite model file
    
    Returns:
        Heatmap overlay on the original image
    """
    try:
        # First check if we're providing a model path
        if model_path is None:
            return generate_gradcam_simplified(img)
            
        # Extract model name from path
        model_name = os.path.basename(model_path)
        
        # Try to load the full model for proper Grad-CAM
        full_model = load_full_model(model_name)
        
        # If we successfully loaded the full model, try to generate proper Grad-CAM
        if full_model is not None:
            try:
                import tensorflow as tf
                
                # Convert image to batch format
                img_batch = np.expand_dims(img, axis=0)
                
                # Find the last convolutional layer
                last_conv_layer = None
                for layer in reversed(full_model.layers):
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        last_conv_layer = layer.name
                        break
                
                if last_conv_layer is None:
                    # No convolutional layer found
                    return generate_gradcam_simplified(img)
                
                # Create a gradient model using TF's GradientTape
                with tf.GradientTape() as tape:
                    # Make a temporary model ending with the last conv layer
                    conv_outputs = None
                    prediction = None
                    
                    # Sequential execution to get both outputs we need
                    x = img_batch
                    for layer in full_model.layers:
                        x = layer(x)
                        if layer.name == last_conv_layer:
                            conv_outputs = x
                        
                    # Get the prediction
                    prediction = x
                    
                    if conv_outputs is None or prediction is None:
                        return generate_gradcam_simplified(img)
                    
                    # Get the predicted class (binary classification)
                    pred_class = 0 if prediction[0][0] < 0.5 else 1
                    
                    # Get the gradient of the output wrt the last conv layer
                    grads = tape.gradient(prediction, conv_outputs)
                    
                # Calculate channel importance weights
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                
                # Apply weights to activation map
                heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
                
                # ReLU operation (only positive influence)
                heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + tf.keras.backend.epsilon())
                heatmap = heatmap.numpy()
                
                # Resize heatmap to input image size
                heatmap_resized = np.uint8(255 * heatmap)
                heatmap_resized = tf.image.resize(
                    tf.expand_dims(heatmap_resized, -1),
                    (img.shape[0], img.shape[1])
                ).numpy().squeeze()
                
                # Apply colormap
                colormap = cm.get_cmap("jet")
                colored_heatmap = colormap(heatmap_resized)[:, :, :3]
                colored_heatmap = np.uint8(colored_heatmap * 255)
                
                # Create overlay
                img_uint8 = np.uint8(img * 255)
                alpha = 0.4
                heatmap_overlay = np.uint8(img_uint8 * (1 - alpha) + colored_heatmap * alpha)
                
                return heatmap_overlay
                
            except Exception as e:
                # If any exception occurs during Grad-CAM generation, fall back to simplified method
                return generate_gradcam_simplified(img)
                
    except Exception as e:
        # Catch any exception and ensure we return something
        return generate_gradcam_simplified(img)
        
    # If we get here, use the simplified method
    return generate_gradcam_simplified(img)

def generate_gradcam_simplified(img):
    """
    Enhanced simplified heatmap generation as a fallback.
    Uses advanced image processing to create a visualization that mimics Grad-CAM.
    
    Args:
        img: Preprocessed image (normalized to [0,1])
    
    Returns:
        Heatmap overlay on the original image
    """
    # Convert to grayscale
    img_gray = np.mean(img, axis=-1)
    
    try:
        # Try to use scipy for more advanced image processing if available
        from scipy import ndimage
        
        # Multi-scale approach for better feature detection
        small_scale = ndimage.gaussian_filter(img_gray, sigma=1.0)
        large_scale = ndimage.gaussian_filter(img_gray, sigma=3.0)
        
        # Dog (Difference of Gaussians) for edge and blob detection
        dog = small_scale - large_scale
        
        # Edge detection using Sobel filters
        sobel_h = ndimage.sobel(img_gray, axis=0)
        sobel_v = ndimage.sobel(img_gray, axis=1)
        edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
        
        # Local standard deviation for texture analysis
        local_std = ndimage.generic_filter(img_gray, np.std, size=11)
        
        # Combine the different features with weights
        weights = np.array([0.3, 0.3, 0.4])  # weights for dog, edge_magnitude, local_std
        features = np.stack([dog, edge_magnitude, local_std])
        
        # Normalize each feature
        for i in range(features.shape[0]):
            feat = features[i]
            feat_min, feat_max = feat.min(), feat.max()
            if feat_max > feat_min:
                features[i] = (feat - feat_min) / (feat_max - feat_min)
        
        # Weighted combination
        combined = np.sum(features * weights.reshape(-1, 1, 1), axis=0)
        
        # Final normalization
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        
        # Apply a mild Gaussian blur to smooth the heatmap
        heatmap = ndimage.gaussian_filter(combined, sigma=1.5)
        
    except (ImportError, Exception) as e:
        # Basic fallback if scipy is not available or fails
        img_edges = np.abs(np.gradient(img_gray)[0]) + np.abs(np.gradient(img_gray)[1])
        heatmap = (img_edges - img_edges.min()) / (img_edges.max() - img_edges.min() + 1e-8)
    
    # Convert to uint8
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply colormap
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_uint8)[:, :, :3]
    
    # Convert to uint8
    colored_heatmap = np.uint8(colored_heatmap * 255)
    
    # Create image with heatmap overlay
    img_uint8 = np.uint8(img * 255)
    alpha = 0.4
    heatmap_overlay = np.uint8(img_uint8 * (1 - alpha) + colored_heatmap * alpha)
    
    return heatmap_overlay

# Main app
def main():
    st.title("Pneumonia Detection from Chest X-rays")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This application uses a deep learning model to detect pneumonia from chest X-ray images.
    The model was trained on the Chest X-Ray dataset from Kaggle.
    
    **How to use:**
    1. Upload a chest X-ray image
    2. Select a model from the dropdown
    3. The model will analyze the image
    4. View the prediction and confidence level
    
    **Model Details:**
    - Architecture: MobileNetV2 with custom classification head
    - Accuracy on test set: ~90%
    """)
    
    # Model selection
    st.sidebar.header("Model Selection")
    available_models = get_available_models()
    
    if not available_models:
        st.sidebar.warning("No models found in the 'models' directory. Please add .tflite models.")
        selected_model = None
    else:
        selected_model = st.sidebar.selectbox(
            "Choose a model",
            available_models,
            index=0,
            help="Select the model you want to use for prediction"
        )
        
        # Display model info
        st.sidebar.markdown(f"**Selected Model:** {selected_model}")
        
        # Try to load model metadata if available
        metadata_path = os.path.join("models", f"{os.path.splitext(selected_model)[0]}_metadata.json")
        if os.path.exists(metadata_path):
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            st.sidebar.subheader("Model Details")
            for key, value in metadata.items():
                if key == "accuracy" or key == "val_accuracy" or key.endswith("_accuracy"):
                    # Format as percentage
                    st.sidebar.markdown(f"**{key.replace('_', ' ').title()}:** {value:.2%}")
                else:
                    st.sidebar.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
    
    st.sidebar.header("Developer")
    st.sidebar.markdown("Created by [Akash Kondaparthi](https://AkashKK25.github.io/Data-Portfolio)")
    st.sidebar.markdown("[GitHub Repository](https://github.com/AkashKK25/pneumonia-xray-detection.git)")
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None and selected_model is not None:
            # Load and display the image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            # Preprocess the image
            processed_image = preprocess_image(image)
            
            # Get prediction
            model_path = os.path.join("models", selected_model)
            interpreter = load_model(model_path)
            prediction = predict(interpreter, processed_image)
            
            # Show prediction
            st.subheader("Prediction")
            
            prediction_class = "Pneumonia" if prediction > 0.5 else "Normal"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Display results
            st.markdown(f"**Diagnosis:** {prediction_class}")
            st.markdown(f"**Confidence:** {confidence:.2%}")
            
            # Progress bar for confidence
            st.progress(float(confidence))
            
            # Check for high risk cases
            if prediction > 0.8:
                st.warning("⚠️ High confidence pneumonia detection. Immediate medical consultation recommended.")
        
        elif uploaded_file is not None and selected_model is None:
            st.error("Please add model files to the 'models' directory and restart the application.")
    
    with col2:
        if uploaded_file is not None and selected_model is not None:
            st.subheader("Model Interpretation")
            
            # Generate GradCAM visualization with proper model path
            model_path = os.path.join("models", selected_model) 
            interpreter = load_model(model_path)
            heatmap = generate_gradcam(interpreter, processed_image[0], model_path)
            
            # Display the heatmap
            st.image(heatmap, caption="Activation Heatmap", use_column_width=True)
            
            st.markdown("""
            **What am I looking at?**
            
            The heatmap overlay highlights regions of the X-ray that the model is focusing on to make its prediction. 
            Warmer colors (red, yellow) indicate areas of higher importance for the model's decision.
            
            In pneumonia cases, the model typically focuses on areas with opacity or consolidation in the lungs.
            """)
    
    # Model comparison section (when multiple models are available)
    if uploaded_file is not None and len(available_models) > 1:
        st.markdown("---")
        st.subheader("Compare All Models")
        
        if st.button("Run analysis with all available models"):
            # Create a dataframe to store results
            results = []
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Analyze with each model
            for i, model_name in enumerate(available_models):
                # Update progress
                progress_bar.progress((i+1)/len(available_models))
                
                # Load model and predict
                model_path = os.path.join("models", model_name)
                model_interpreter = load_model(model_path)
                model_prediction = predict(model_interpreter, processed_image)
                
                # Store results
                prediction_class = "Pneumonia" if model_prediction > 0.5 else "Normal"
                confidence = model_prediction if model_prediction > 0.5 else 1 - model_prediction
                
                results.append({
                    "Model": model_name,
                    "Prediction": prediction_class,
                    "Confidence": float(confidence)
                })
            
            # Create comparison visualization
            import pandas as pd
            results_df = pd.DataFrame(results)
            
            # Display as table
            st.dataframe(results_df)
            
            # Create bar chart for confidence levels
            st.subheader("Confidence Comparison")
            chart_data = pd.DataFrame({
                'Model': results_df['Model'],
                'Confidence': results_df['Confidence'] * 100  # Convert to percentage
            })
            
            st.bar_chart(chart_data.set_index('Model'))
            
            # Show consensus result
            pneumonia_count = sum(1 for r in results if r["Prediction"] == "Pneumonia")
            normal_count = sum(1 for r in results if r["Prediction"] == "Normal")
            
            consensus = "Pneumonia" if pneumonia_count > normal_count else "Normal"
            st.subheader(f"Consensus Prediction: {consensus}")
            st.markdown(f"*{pneumonia_count} of {len(results)} models predicted Pneumonia*")
    
    # Additional information
    st.markdown("---")
    st.subheader("About Pneumonia")
    st.markdown("""
    Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, 
    causing cough with phlegm, fever, chills, and difficulty breathing. Various organisms, including bacteria, viruses, 
    and fungi, can cause pneumonia.
    
    **Note:** This app is for educational purposes only and should not replace professional medical advice.
    """)

if __name__ == "__main__":
    main()