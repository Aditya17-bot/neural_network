import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.network import SimpleCNN
import glob
import os

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model():
    # Get the latest model
    model_files = glob.glob('models/model_*.pth')
    latest_model = max(model_files, key=os.path.getctime)
    
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    model.eval()
    
    # Test 1: Check model parameters
    print("Testing model parameters...")
    param_count = count_parameters(model)
    assert param_count < 100000, f"Model has {param_count} parameters, should be < 100000"
    print(f"✓ Model has {param_count} parameters")
    
    # Test 2: Check input shape
    print("\nTesting input shape handling...")
    test_input = torch.randn(1, 1, 28, 28).to(device)
    try:
        output = model(test_input)
        assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"
        print("✓ Model handles 28x28 input correctly")
    except Exception as e:
        raise AssertionError(f"Model failed to process 28x28 input: {str(e)}")
    
    # Test 3: Check accuracy
    print("\nTesting model accuracy...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    assert accuracy > 80, f"Accuracy is {accuracy:.2f}%, should be > 80%"
    print(f"✓ Model accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    test_model() 