import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from model import MNISTNet

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Dataset and Dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load exactly 10k test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    
    # 2. Model Loading
    model = MNISTNet().to(device)
    model_path = 'outputs/mnist_model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Please run train.py first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    # 3. Model Inference
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Use forward which only returns out for standard evaluation
            outputs = model(images)
            if isinstance(outputs, tuple):
               outputs = outputs[0]
            
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 4. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("--- Model Evaluation Metrics ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    # 5. Confusion Matrix Calculation and Plotting
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - MNIST ANN')
    plt.ylabel('Actual Digit')
    plt.xlabel('Predicted Digit')
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png')
    print("Confusion Matrix plot saved at outputs/confusion_matrix.png")

if __name__ == '__main__':
    evaluate()
