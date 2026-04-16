import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from model import MNISTNet

st.set_page_config(page_title="MNIST ANN Cloud Deployment", layout="wide")

# Constants and Parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4

@st.cache_resource
def load_model():
    model = MNISTNet()
    model_path = 'outputs/mnist_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    else:
        return None

def process_image(image):
    # Convert image to grayscale (MNIST is 1 channel)
    image = image.convert('L')
    # Resize to 28x28 if not already
    image = image.resize((28, 28))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Needs to be shape [1, 1, 28, 28] for the network
    tensor_image = transform(image).unsqueeze(0)
    return tensor_image

def main():
    st.title("MNIST ANN Analysis & Cloud Deployment")
    
    st.sidebar.header("Architecture & Parameters")
    st.sidebar.markdown(f"**Dataset**: MNIST (28x28 Handwritten Images)")
    st.sidebar.markdown(f"**Training Split**: 50,00k | **Testing**: 10k")
    st.sidebar.markdown(f"**Learning Rate**: 10^{int(np.log10(LEARNING_RATE))}")
    st.sidebar.markdown(f"**Batch Size**: {BATCH_SIZE}")
    
    st.sidebar.subheader("Hidden Layers Architecture")
    st.sidebar.markdown("""
    - **Layer 1**: 64 Units (ReLU)
    - **Layer 2**: 64 Units (ReLU)
    - **Layer 3**: 32 Units (ReLU)
    - **Output**: 10 Units
    """)
    
    st.write("### Upload Image for Prediction")
    
    model = load_model()
    if model is None:
        st.error("Model weights 'outputs/mnist_model.pth' not found. Please train the model locally first.")
        return

    uploaded_file = st.file_uploader("Browse / Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Uploaded Image', width=150)
            
        tensor_image = process_image(image)
        
        # Inference
        with torch.no_grad():
            outputs, features = model.forward_features(tensor_image)
            _, predicted = torch.max(outputs.data, 1)
            pred_class = predicted.item()
            
            # Use softmax for probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
            
        with col2:
            st.success(f"## Prediction: {pred_class}")
            st.write(f"**Confidence**: {probs[pred_class]*100:.2f}%")
        
        st.write("---")
        st.write("### Layer-wise Features Visualization")
        
        st.write("Displaying intermediate activations of the hidden layers:")
        
        # Format function for reshaped activations
        # layer1: 64 -> 8x8
        # layer2: 64 -> 8x8
        # layer3: 32 -> 8x4 (padding to 8x8 or just show as 1D/2D)
        
        f1 = features['layer1'].squeeze().numpy().reshape(8, 8)
        f2 = features['layer2'].squeeze().numpy().reshape(8, 8)
        f3_flat = features['layer3'].squeeze().numpy()
        # Reshape 32 into 4x8
        f3 = f3_flat.reshape(4, 8)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Layer 1
        im1 = axes[0].imshow(f1, cmap='viridis')
        axes[0].set_title('Layer 1 (64 units) -> 8x8')
        axes[0].axis('off')
        fig.colorbar(im1, ax=axes[0], shrink=0.5)
        
        # Plot Layer 2
        im2 = axes[1].imshow(f2, cmap='viridis')
        axes[1].set_title('Layer 2 (64 units) -> 8x8')
        axes[1].axis('off')
        fig.colorbar(im2, ax=axes[1], shrink=0.5)

        # Plot Layer 3
        im3 = axes[2].imshow(f3, cmap='viridis')
        axes[2].set_title('Layer 3 (32 units) -> 4x8')
        axes[2].axis('off')
        fig.colorbar(im3, ax=axes[2], shrink=0.5)
        
        st.pyplot(fig)

if __name__ == '__main__':
    main()
