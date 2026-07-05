import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletFrequencySplitLayer(nn.Module):
    def __init__(self,in_channels):
        super(WaveletFrequencySplitLayer,self).__init__()
        self.in_channels=in_channels
        #Haar Wavelets:
        ll_wavelet=torch.tensor([[0.5,0.5],[0.5,0.5]])
        lh_wavelet=torch.tensor([[0.5,0.5],[-0.5,0.5]])
        hl_wavelet=torch.tensor([[0.5,-0.5],[0.5,-0.5]])
        hh_wavelet=torch.tensor([[0.5,-0.5],[-0.5,0.5]])

        filters=torch.stack([ll_wavelet,lh_wavelet,hl_wavelet,hh_wavelet],dim=0).unsqueeze(1)
        filters=filters.repeat(in_channels,1,1,1)

        self.register_buffer('weight',filters)

    def forward(self,x):

        out=F.conv2d(x,self.weight,stride=2,padding=0,groups=self.in_channels)
        n,c_out,h,w=out.shape
        out=out.view(n,self.in_channels,4,h,w)
        #Wavelet Coefficients:
        ll_coeff=out[:,:,0,:,:]
        lh_coeff=out[:,:,1,:,:]
        hl_coeff = out[:, :, 2, :, :]
        hh_coeff = out[:, :,3,:,:]

        return ll_coeff,lh_coeff,hl_coeff,hh_coeff

class convolutionalNeuralNetwork(nn.Module):
    def __init__(self,num_classes=10):
        super(convolutionalNeuralNetwork,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(32)

        self.wavelet_pool1 = WaveletFrequencySplitLayer(in_channels=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.wavelet_pool2=WaveletFrequencySplitLayer(in_channels=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        ll, lh1, hl1, hh1 = self.wavelet_pool1(x)

        # Layer 2 (processing the approximation sub-band)
        x = F.relu(self.bn2(self.conv2(ll)))

        # Wavelet Decomposition 2
        ll, lh2, hl2, hh2 = self.wavelet_pool2(x)

        # Layer 3
        x = F.relu(self.bn3(self.conv3(ll)))

        # Classification head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


import torch.optim as optim
import torch.nn as nn

# 1. Initialize the model
model = convolutionalNeuralNetwork(num_classes=10)

# 2. Define the FEEDBACK criteria (Loss Function and Optimizer)
criterion = nn.CrossEntropyLoss()  # Measures the error
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Tweaks the weights based on error

# --- The Feedback Loop (Repeated thousands of times) ---
for epoch in range(10):
    # Setup dummy data and the ACTUAL correct targets (e.g., Image 1 is a Cat)
    dummy_input = torch.randn(4, 3, 32, 32)
    true_labels = torch.tensor([3, 0, 5, 2])  # The real objects

    # STEP A: One-way forward pass (Matrix to Matrix)
    outputs = model(dummy_input)

    # STEP B: Calculate the error
    loss = criterion(outputs, true_labels)

    # STEP C: THE FEEDBACK STAGE
    optimizer.zero_grad()  # Clear old feedback scores
    loss.backward()  # Calculate the error gradients backward through the network
    optimizer.step()  # Physically adjust the wave parameters to fix the error

    print(f"Epoch {epoch} - Error Level: {loss.item():.4f}")

# Create a dummy batch: 4 images, 3 channels (RGB), 32x32 resolution (e.g., CIFAR-10 size)
dummy_input = torch.randn(4, 3, 32, 32)

# Initialize model
model = convolutionalNeuralNetwork(num_classes=10)

# Forward pass
output = model(dummy_input)

print(f"Input batch shape:   {dummy_input.shape}")
print(f"Output batch shape:  {output.shape}")
print("\nModel Architecture Verification Successful!")