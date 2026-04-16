import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
from model import MNISTNet

def main():
    # Set hyper-parameters correctly according to requirements
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    
    # 1. Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Dataset and Dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean/std
    ])
    
    # Download the whole MNIST train set
    train_dataset_full = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    
    # The requirement specifically limits training to the first 50,000 samples
    train_subset = Subset(train_dataset_full, range(50000))
    
    train_loader = DataLoader(dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Model instantiation
    model = MNISTNet().to(device)
    
    # 4. Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training loop
    model.train()
    epoch_losses = []
    
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")
        
    # 6. Save the trained model
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/mnist_model.pth')
    print("Model saved to outputs/mnist_model.pth")
    
    # 7. Plot Loss vs Epochs and save
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', linestyle='-', color='b')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.savefig('outputs/loss_vs_epoch.png')
    print("Loss vs Epoch plot saved at outputs/loss_vs_epoch.png")

if __name__ == '__main__':
    main()
