import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class DimensionalityEstimatorNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DimensionalityEstimatorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Changed from 784 to input_size
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the base dimensions
base_dimensions = torch.tensor([
    90,  # Metric Tensor
    4,   # Gravitational Wave
    2,   # Polarization
    3,   # Quadrupole
    5,   # Energy Emission
    30    # Binary Waveform
], dtype=torch.float32)

# Additional contributions from Fibonacci-driven layers
fib_mod7_unique = 7  # Placeholder for len(set(fib_mod7))
fib_mod9_unique = 9  # Placeholder for len(set(fib_mod9))
digital_root_unique = 5  # Placeholder for unique digital root values

additional_dimensions = torch.tensor([
    fib_mod7_unique, 
    fib_mod9_unique, 
    digital_root_unique
], dtype=torch.float32)

# Combine all dimensions
input_data = torch.cat((base_dimensions, additional_dimensions))

# Instantiate the neural network
input_size = len(input_data)  # Changed to match input data size
hidden_size = 16  # Hidden layer size
output_size = 1   # Output a single estimated dimensionality

model = DimensionalityEstimatorNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Generate a dummy target (assumed total dimensionality)
target = torch.tensor([sum(input_data)], dtype=torch.float32)

# Train the model for a few iterations
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Get the final estimated dimensionality
estimated_dimensionality = model(input_data).item()
print(f'Estimated Total Dimensionality: {estimated_dimensionality:.2f}')
# Calculate accuracy metrics
actual_dim = target.item()
predicted_dim = estimated_dimensionality
abs_error = abs(actual_dim - predicted_dim)
rel_error = (abs_error / actual_dim) * 100

print("\nAccuracy Metrics:")
print("-" * 50)
print(f"Actual Dimensionality: {actual_dim:.2f}")
print(f"Predicted Dimensionality: {predicted_dim:.2f}")
print(f"Absolute Error: {abs_error:.2f}")
print(f"Relative Error: {rel_error:.2f}%")

# Calculate accuracy score (as percentage)
accuracy = 100 - rel_error
print(f"Model Accuracy: {accuracy:.2f}%")

# Visualize accuracy range
print("\nAccuracy Range:")
print("-" * 50)
range_width = 50
position = int((accuracy / 100) * range_width)
accuracy_bar = "[" + "=" * position + ">" + " " * (range_width - position - 1) + "]"
print(accuracy_bar)
print(f"0%{' ' * (range_width-6)}100%")
# Add MNIST training section
print("\nTraining on MNIST Dataset:")
print("-" * 50)

# Import required libraries
from torchvision import transforms, datasets

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Create a new model instance for MNIST with correct input size
mnist_model = DimensionalityEstimatorNN(784, hidden_size, output_size)
optimizer = optim.Adam(mnist_model.parameters(), lr=0.01)

# Train on MNIST
mnist_epochs = 5
for epoch in range(mnist_epochs):
    mnist_model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = mnist_model(data.view(data.size(0), -1))  # Flatten the images
        loss = criterion(output, target.float())
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'MNIST Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    # Test accuracy after each epoch
    mnist_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = mnist_model(data.view(data.size(0), -1))
            test_loss += criterion(output, target.float()).item()
            pred = output.round()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'MNIST Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
