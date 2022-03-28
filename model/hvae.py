import torch
from torch import nn
from torch.nn import functional as F

# %%
"""
VAE_BN: HVAE Architecture with seperated decoder structure
VAE_NoBN: HVAE Architecture with not seperated decoder structure
"""
class VAE_BN(nn.Module):
    def __init__(self, layers, embedding_dim):
        super(VAE_BN, self).__init__()

        self.layers = layers
        self.embedding_dim = embedding_dim

        self.fc1 = nn.Linear(layers[0], layers[1])
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(layers[3], embedding_dim * 2)
        self.fc51 = nn.Linear(embedding_dim, layers[3])
        self.dropout51 = nn.Dropout(0.1)
        self.fc52 = nn.Linear(layers[3], layers[2])
        self.dropout52 = nn.Dropout(0.1)
        self.fc53 = nn.Linear(layers[2], layers[1])
        self.dropout53 = nn.Dropout(0.1)
        self.fc54 = nn.Linear(layers[1], layers[0])
        self.fc61 = nn.Linear(embedding_dim, layers[3])
        self.dropout61 = nn.Dropout(0.1)
        self.fc62 = nn.Linear(layers[3], layers[2])
        self.dropout62 = nn.Dropout(0.1)
        self.fc63 = nn.Linear(layers[2], layers[1])
        self.dropout63 = nn.Dropout(0.1)
        self.fc64 = nn.Linear(layers[1], layers[0])
        self.soft_plus = nn.Softplus()

    def encode(self, x):
        output = F.elu(self.fc1(x))
        output = self.dropout1(output)
        output = F.elu(self.fc2(output))
        output = self.dropout2(output)
        output = F.elu(self.fc3(output))
        output = self.dropout3(output)
        output = self.fc4(output)
        return output[:,:self.embedding_dim], output[:,self.embedding_dim:self.embedding_dim*2]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        output1 = F.elu(self.fc51(z))
        output1 = self.dropout51(output1)
        output1 = F.elu(self.fc52(output1))
        output1 = self.dropout52(output1)
        output1 = F.elu(self.fc53(output1))
        output1 = self.dropout53(output1)
        output1 = self.fc54(output1)
        
        output2 = F.elu(self.fc61(z))
        output2 = self.dropout61(output2)
        output2 = F.elu(self.fc62(output2))
        output2 = self.dropout62(output2)
        output2 = F.elu(self.fc63(output2))
        output2 = self.dropout63(output2)
        output2 = self.fc64(output2)
        return output1, self.soft_plus(output2)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat, sigma_hat = self.decode(z)
        return x_hat, sigma_hat, mu, logvar

# %%
class VAE_NoBN(nn.Module):
    def __init__(self, layers, embedding_dim):
        super(VAE_NoBN, self).__init__()

        self.layers = layers
        self.embedding_dim = embedding_dim

        self.fc1 = nn.Linear(layers[0], layers[1])
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(layers[3], embedding_dim * 2)
        self.fc51 = nn.Linear(embedding_dim, layers[3])
        self.dropout51 = nn.Dropout(0.1)
        self.fc52 = nn.Linear(layers[3], layers[2])
        self.dropout52 = nn.Dropout(0.1)
        self.fc53 = nn.Linear(layers[2], layers[1])
        self.dropout53 = nn.Dropout(0.1)
        self.fc54 = nn.Linear(layers[1], layers[0] * 2)
        self.soft_plus = nn.Softplus()

        
    def encode(self, x):
        output = F.elu(self.fc1(x))
        output = self.dropout1(output)
        output = F.elu(self.fc2(output))
        output = self.dropout2(output)
        output = F.elu(self.fc3(output))
        output = self.dropout3(output)
        output = self.fc4(output)
        return output[:, :self.embedding_dim], output[:, self.embedding_dim:self.embedding_dim*2]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        output1 = F.elu(self.fc51(z))
        output1 = self.dropout51(output1)
        output1 = F.elu(self.fc52(output1))
        output1 = self.dropout52(output1)
        output1 = F.elu(self.fc53(output1))
        output1 = self.dropout53(output1)
        output1 = self.fc54(output1)
        return output1[:, :self.layers[0]], self.soft_plus(output1[:, self.layers[0]:self.layers[0]*2])

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat, sigma_hat = self.decode(z)
        return x_hat, sigma_hat, mu, logvar


def weighted_mse_loss(input, target, weight, eps = 0.0001):
    return torch.mean(torch.reciprocal(weight + eps) * ((input - target) ** 2), dim = 1)

# %%
def loss_function(beta, recon_x, recon_sigma, x, mu, logvar, eps = 0.0001):
    Recon_error = weighted_mse_loss(recon_x, x, recon_sigma, eps) + torch.sum(torch.abs(torch.log(recon_sigma + eps)), dim = 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    loss = (beta * Recon_error) + KLD
    return loss.mean()

def loss_function_test(beta, recon_x, recon_sigma, x, mu, logvar, eps = 0.0001):
    Recon_error = weighted_mse_loss(recon_x, x, recon_sigma, eps) + torch.sum(torch.abs(torch.log(recon_sigma + eps)), dim = 1)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    loss = (beta * Recon_error) + KLD
    return loss
