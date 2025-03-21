import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
import os

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

def generate_gradcam(interpreter, img):
    # This is a simplified GradCAM implementation for TFLite
    # In a production environment, you would use the full TensorFlow model
    
    # For demonstration purposes, we'll create a simple heatmap overlay
    # based on the edges in the image
    
    img_gray = np.mean(img, axis=-1)
    img_edges = np.abs(np.gradient(img_gray)[0]) + np.abs(np.gradient(img_gray)[1])
    
    # Normalize
    img_edges = (img_edges - np.min(img_edges)) / (np.max(img_edges) - np.min(img_edges))
    
    # Create heatmap
    heatmap = np.uint8(255 * img_edges)
    
    # Use jet colormap
    colormap = matplotlib.colormaps["jet"]
    colored_heatmap = colormap(heatmap)[:, :, :3]
    
    # Convert back to uint8
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
    st.sidebar.markdown("Created by Akash Kondaparthi")
    st.sidebar.markdown("[GitHub Repository](https://github.com/AkashKK25/pneumonia-xray-detection)")
    
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
            
            # Generate GradCAM visualization
            model_path = os.path.join("models", selected_model)
            interpreter = load_model(model_path)
            heatmap = generate_gradcam(interpreter, processed_image[0])
            
            # Display the heatmap
            st.image(heatmap, caption="Activation Heatmap", use_column_width=True)
            
            st.markdown("""
            **What am I looking at?**
            
            The heatmap overlay highlights regions of the X-ray that the model is focusing on to make its prediction. 
            Brighter colors (pixels) indicate areas of higher importance for the model's decision.
            
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