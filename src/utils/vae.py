import torch 



class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, beta =1.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_log_var = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_log_var[:, 0, :]
        log_var = mu_log_var[:, 1, :]
        z = self.sample(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var, z

    def loss(self, s):
        # compute reconstruction loss
        x_recon, mu, log_var, z = self(s)
        recon_loss = torch.nn.functional.mse_loss(x_recon, s, reduction='sum')
        # compute kl divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kl_div
    
    def loss_n(self, s):
        # compute reconstruction loss
        x_recon, mu, log_var, z = self(s)
    
        recon_loss =torch.sqrt( torch.sum(torch.nn.functional.mse_loss(x_recon, s, reduction='none'),dim=-1))
        # compute kl divergence
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return recon_loss + kl_div

    def sample_latent(self, x):
        mu_log_var = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_log_var[:, 0, :]
        log_var = mu_log_var[:, 1, :]
        z = self.sample(mu, log_var)
        return z

    def sample_recon(self, z):
        x_recon = self.decoder(z)
        return x_recon

    def sample_latent_recon(self, x):
        mu_log_var = self.encoder(x).view(-1, 2, self.latent_dim)
        mu = mu_log_var[:, 0, :]
        log_var = mu_log_var[:, 1, :]
        z = self.sample(mu, log_var)
        x_recon = self.decoder(z)
        return z, x_recon