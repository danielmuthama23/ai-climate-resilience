import torch
import torch.nn as nn
import numpy as np

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, output_dim: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, z):
        return self.model(z)

def train_gan(data_path: str, epochs: int = 1000, batch_size: int = 32):
    """Train GAN to augment sensor data."""
    real_data = torch.FloatTensor(np.load(os.path.join(data_path, "sensor_data.npy")))
    
    # Initialize models and optimizers
    generator = Generator()
    discriminator = nn.Sequential(nn.Linear(2, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())
    opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    
    # Training loop
    for epoch in range(epochs):
        # Train discriminator
        z = torch.randn(batch_size, 100)
        fake_data = generator(z)
        real_loss = torch.log(discriminator(real_data)).mean()
        fake_loss = torch.log(1 - discriminator(fake_data.detach())).mean()
        loss_d = -(real_loss + fake_loss)
        
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        
        # Train generator
        loss_g = -torch.log(discriminator(fake_data)).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.stup()

    # Save synthetic data
    z = torch.randn(1000, 100)
    synthetic_data = generator(z).detach().numpy()
    np.save(os.path.join(data_path, "../synthetic/synthetic_sensor.npy"), synthetic_data)