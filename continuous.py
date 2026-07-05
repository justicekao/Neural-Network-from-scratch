import torch
import torch.nn as nn
import numpy as np


class ContinuousWaveletConv2d(nn.Module):
    """
    A Convolutional layer constrained by continuous Gabor Wavelet functions.
    Instead of learning thousands of individual pixel weights, it learns
    the spatial frequency, orientation, and scale parameters of continuous waves.
    """

    def __init__(self, in_channels, out_channels, kernel_size=15):
        super(ContinuousWaveletConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Ensure kernel size is odd for symmetric padding
        assert kernel_size % 2 == 1, "Kernel size must be odd."

        # Define a spatial grid for the continuous wave equations (-X to +X)
        radius = kernel_size // 2
        y, x = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))

        # Register the spatial grid coordinates as static buffers
        self.register_buffer('grid_x', torch.FloatTensor(x))
        self.register_buffer('grid_y', torch.FloatTensor(y))

        # Initialize continuous wave parameters as learnable PyTorch parameters
        # Theta (Orientation): Distributed uniformly between 0 and Pi
        thetas = np.linspace(0, np.pi, out_channels, endpoint=False)
        self.theta = nn.Parameter(torch.FloatTensor(thetas))

        # Sigma (Envelope variance/scale)
        self.sigma = nn.Parameter(torch.ones(out_channels) * (kernel_size / 4))

        # Frequency (Wave speed)
        self.frequency = nn.Parameter(torch.ones(out_channels) * (2.0 * np.pi / kernel_size))

        # Phase offset
        self.psi = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        # Dynamically build the continuous wavelet kernels based on current parameters
        kernels = []

        for i in range(self.out_channels):
            # Rotate coordinates based on learned wave orientation
            x_rot = self.grid_x * torch.cos(self.theta[i]) + self.grid_y * torch.sin(self.theta[i])
            y_rot = -self.grid_x * torch.sin(self.theta[i]) + self.grid_y * torch.cos(self.theta[i])

            # Continuous Gaussian Envelope (The bounding curve)
            gaussian = torch.exp(-0.5 * (x_rot ** 2 + y_rot ** 2) / (self.sigma[i] ** 2 + 1e-6))

            # Continuous Sinusoidal Wave (The oscillating wave)
            sinusoid = torch.cos(self.frequency[i] * x_rot + self.psi[i])

            # Combine to get the continuous Gabor Wavelet
            gabor_kernel = gaussian * sinusoid

            # Normalize the kernel to maintain stable scale activations
            gabor_kernel = gabor_kernel / (gabor_kernel.norm(p=2) + 1e-6)
            kernels.append(gabor_kernel)

        # Stack into shape: [out_channels, 1, kernel_size, kernel_size]
        stacked_kernels = torch.stack(kernels).unsqueeze(1)

        # Repeat across input channels (e.g., RGB) to allow standard convolution
        # Shape becomes: [out_channels, in_channels, kernel_size, kernel_size]
        weight = stacked_kernels.repeat(1, self.in_channels, 1, 1)

        # Apply standard convolution using our generated continuous wave kernels
        padding = self.kernel_size // 2
        return torch.nn.functional.conv2d(x, weight, padding=padding)


# --- Example Usage inside a CNN ---
class ContinuousWaveletCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ContinuousWaveletCNN, self).__init__()

        # Layer 1: Automatically extracts continuous wave patterns at 16 different angles/scales
        self.wavelet_layer = ContinuousWaveletConv2d(in_channels=3, out_channels=16, kernel_size=15)
        self.bn1 = nn.BatchNorm2d(16)

        # Subsequent standard layers process the continuous wave feature maps
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 16 * 16, num_classes)  # Assuming 32x32 input size

    def forward(self, x):
        x = torch.relu(self.bn1(self.wavelet_layer(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    dummy_img = torch.randn(2, 3, 32, 32)  # 2 images, RGB, 32x32
    model = ContinuousWaveletCNN()
    output = model(dummy_img)
    print("Output shape:", output.shape)  # Output shape: torch.Size([2, 10])
