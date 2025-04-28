import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, input):
        return self.main(input)

class WGAN:
    def __init__(self, data_loader):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.data_loader = data_loader
        self.optimizer_g = optim.RMSprop(self.generator.parameters(), lr=0.00005)
        self.optimizer_d = optim.RMSprop(self.discriminator.parameters(), lr=0.00005)
        self.epochs = 0
        self.losses = []

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for i, data in enumerate(self.data_loader, 0):
                real_data = data[0]
                batch_size = real_data.size(0)

                # Train Discriminator
                self.optimizer_d.zero_grad()
                output_real = self.discriminator(real_data)
                noise = torch.randn(batch_size, 100)
                fake_data = self.generator(noise).detach()
                output_fake = self.discriminator(fake_data)
                loss_d = -(torch.mean(output_real) - torch.mean(output_fake))
                loss_d.backward()
                self.optimizer_d.step()

                # Train Generator
                self.optimizer_g.zero_grad()
                noise = torch.randn(batch_size, 100)
                fake_data = self.generator(noise)
                output_fake = self.discriminator(fake_data)
                loss_g = -torch.mean(output_fake)
                loss_g.backward()
                self.optimizer_g.step()

                # Save losses
                self.losses.append((loss_d.item(), loss_g.item()))

            self.epochs += 1
            self.save_checkpoint()

    def save_checkpoint(self):
        checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'epochs': self.epochs,
            'losses': self.losses
        }
        torch.save(checkpoint, f'checkpoint_epoch_{self.epochs}.pth')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.epochs = checkpoint['epochs']
        self.losses = checkpoint['losses']

def main():
    # Load data
    data_loader = DataLoader(...)  # Replace with actual data loader

    # Initialize WGAN
    wgan = WGAN(data_loader)

    # Load checkpoint if available
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        wgan.load_checkpoint(checkpoint_path)

    # Train WGAN
    wgan.train(num_epochs=100)

    # Save final checkpoint
    wgan.save_checkpoint()

    # Return loss metrics
    with open('loss_metrics.json', 'w') as f:
        json.dump(wgan.losses, f)

if __name__ == '__main__':
    main()
