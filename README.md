# MNIST ANN Cloud Deployment

This project deploys a PyTorch Artificial Neural Network (ANN) trained on the MNIST dataset as a Streamlit web application. 

## Features
- Upload hand-drawn digits (images) to be classified by the neural network.
- Live inference and prediction output with confidence score using softmax.
- Visualizes layer-wise activations (hidden layers) to understand how the network processes the image.

## Architecture
The neural network (`MNISTNet`) is defined with 3 hidden layers:
- Layer 1: 64 units (ReLU)
- Layer 2: 64 units (ReLU)
- Layer 3: 32 units (ReLU)
- Output Layer: 10 units (Digits 0-9)

## Getting Started Locally

1. Create a Python environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. To run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

*(Note: The repository includes a pre-trained model checkpoint in the `outputs/` folder (`mnist_model.pth`), allowing the Streamlit app to run right out of the box without local training.)*

## Deployment on Streamlit Community Cloud
To make the application available online:
1. Push this code to a public GitHub repository. 
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app" and point it to your GitHub repository and branch.
4. Set the "Main file path" to `app.py`.
5. Click **Deploy**. Streamlit will automatically read `requirements.txt` and provision the container!
