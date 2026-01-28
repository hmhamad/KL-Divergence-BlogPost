import matplotlib.pyplot as plt
import numpy as np
import torch
from abc import ABC, abstractmethod

np.random.seed(0)

class Distribution(ABC):
    """
    Abstract base class for probability distributions.
    All distributions must implement sample() and log_prob().
    """
    
    @abstractmethod
    def sample(self, n_samples):
        """Draw n_samples from the distribution."""
        pass
    
    @abstractmethod
    def log_prob(self, x):
        """Compute log probability density at x."""
        pass

def forward_kl(p, q, n_samples=10000):
    """
    Compute KL(P||Q) = E_P[log P(x) - log Q(x)]
    Sample from P, evaluate both P and Q.
    """
    x = p.sample(n_samples)
    return torch.mean(p.log_prob(x) - q.log_prob(x))

def reverse_kl(p, q, n_samples=10000):
    """
    Compute KL(Q||P) = E_Q[log Q(x) - log P(x)]
    Sample from Q, evaluate both Q and P.
    """
    x = q.sample(n_samples)
    return torch.mean(q.log_prob(x) - p.log_prob(x))

class GaussianDistribution(Distribution):
    """Gaussian Distribution for gradient-based optimization."""
    
    def __init__(self, mu, sigma):
        self.mu = torch.as_tensor(mu, dtype=torch.float32).requires_grad_(True)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32).requires_grad_(True)
    
    def sample(self, n_samples):
        """Reparameterization trick: x = mu + sigma * eps"""
        eps = torch.randn(n_samples)
        return self.mu + self.sigma * eps
    
    def log_prob(self, x):
        """Compute log P(x) for torch tensors."""
        return -0.5 * torch.log(2 * torch.pi * self.sigma**2) - 0.5 * (x - self.mu)**2 / self.sigma**2

class GaussianMixtureDistribution(Distribution):
    def __init__(self, means, sigmas, weights):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.weights = self.weights / self.weights.sum()  # normalize
        self.n_components = len(means)
    
    def sample(self, n_samples):
        """Sample from the mixture"""
        # Choose which component each sample comes from
        components = np.random.choice(
            self.n_components, 
            size=n_samples, 
            p=self.weights.numpy()
        )
        # Sample from the chosen components
        samples = np.random.normal(
            self.means.numpy()[components],
            self.sigmas.numpy()[components]
        )
        return torch.tensor(samples, dtype=torch.float32)
    
    def log_prob(self, x):
        """Compute log P(x) using log-sum-exp trick for numerical stability"""
        log_probs = []
        for i in range(self.n_components):
            # Log probability of Gaussian component i
            log_gaussian = -0.5 * torch.log(2 * torch.pi * self.sigmas[i]**2) \
                          - 0.5 * (x - self.means[i])**2 / self.sigmas[i]**2
            # Weight by mixture coefficient
            log_probs.append(torch.log(self.weights[i]) + log_gaussian)
        
        log_probs = torch.stack(log_probs)
        # log(sum(exp(log_probs))) computed stably
        max_log = torch.max(log_probs, dim=0)[0]
        return max_log + torch.log(torch.sum(torch.exp(log_probs - max_log), dim=0))

def optimize_forward_kl(p, mu_init=0.0, sigma_init=1.0, lr=0.1, n_steps=1000, n_samples=1000):
    """
    Optimize Q to minimize KL(P||Q) using gradient descent.
    
    Loss: E_P[-log Q(x)]
    Sample from P (fixed), compute -log Q(x), backprop through Q's parameters.
    """
    # Initialize Q's parameters
    mu = torch.tensor(mu_init, requires_grad=True)
    sigma = torch.tensor(sigma_init, requires_grad=True)
    
    loss_history = []
    
    for step in range(n_steps):
        # Create Q with current parameters
        q = GaussianDistribution(mu, sigma)
        
        # Compute loss: E_P[-log Q(x)]
        loss = forward_kl(p, q, n_samples)
        
        # Backprop
        loss.backward()
        
        # Gradient descent update
        with torch.no_grad():
            mu -= lr * mu.grad
            sigma -= lr * sigma.grad
            
            # Zero gradients
            mu.grad.zero_()
            sigma.grad.zero_()
        
        loss_history.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, mu = {mu.item():.4f}, sigma = {sigma.item():.4f}")
    
    return mu.item(), sigma.item(), loss_history

def optimize_reverse_kl(p, mu_init=0.0, sigma_init=1.0, lr=0.01, n_steps=1000, n_samples=1000):
    """
    Optimize Q to minimize KL(Q||P) using gradient descent.
    
    Loss: E_Q[log Q(x) - log P(x)]
    Sample from Q using reparameterization, compute log Q(x) - log P(x), backprop.
    """
    # Initialize Q's parameters
    mu = torch.tensor(mu_init, requires_grad=True)
    sigma = torch.tensor(sigma_init, requires_grad=True)
    
    loss_history = []
    
    for step in range(n_steps):
        # Create Q with current parameters
        q = GaussianDistribution(mu, sigma)
        
        # Compute loss: E_Q[log Q(x) - log P(x)]
        loss = reverse_kl(p, q, n_samples)
        
        # Backprop
        loss.backward()
        
        # Gradient descent update
        with torch.no_grad():
            mu -= lr * mu.grad
            sigma -= lr * sigma.grad
            
            # Zero gradients
            mu.grad.zero_()
            sigma.grad.zero_()
        
        loss_history.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}, mu = {mu.item():.4f}, sigma = {sigma.item():.4f}")
    
    return mu.item(), sigma.item(), loss_history


def plot_before_after(p, q_init, q_final, title="Optimization Result"):
    """Plot initial and final distributions."""
    x = torch.linspace(-10, 10, 1000)
    
    # Compute densities
    p_vals = torch.exp(p.log_prob(x))
    q_init_vals = torch.exp(q_init.log_prob(x))
    q_final_vals = torch.exp(q_final.log_prob(x))
    
    # Get shared y-axis limits
    all_vals = torch.cat([p_vals, q_init_vals, q_final_vals])
    y_max = all_vals.max().item() * 1.1
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Initial state
    ax1.plot(x, p_vals.detach().numpy(), 'r-', linewidth=2, label='P(x): Target')
    ax1.plot(x, q_init_vals.detach().numpy(), 'b--', linewidth=2,
             label=f'Q_init(x): N({q_init.mu.item():.1f}, {q_init.sigma.item():.1f}²)')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Initial Setup', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, y_max])
    ax1.set_xticks(range(int(x.min().item()), int(x.max().item()) + 1))
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Final state
    ax2.plot(x, p_vals.detach().numpy(), 'r-', linewidth=2, label='P(x): Target')
    ax2.plot(x, q_final_vals.detach().numpy(), 'b-', linewidth=2,
             label=f'Q_final(x): N({q_final.mu.item():.2f}, {q_final.sigma.item():.2f}²)')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('After Optimization', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, y_max])
    ax2.set_xticks(range(int(x.min().item()), int(x.max().item()) + 1))
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=300)
    plt.show()

def compare_forward_reverse(p, mu_fwd, sigma_fwd, mu_rev, sigma_rev):
    """Side-by-side comparison of forward vs reverse KL."""
    x = torch.linspace(-10, 10, 1000)
    
    # P(x): Bimodal mixture
    p_vals = torch.exp(p.log_prob(x))
    
    # Q from forward KL
    q_fwd = GaussianDistribution(mu=mu_fwd, sigma=sigma_fwd)
    q_fwd_vals = torch.exp(q_fwd.log_prob(x))
    
    # Q from reverse KL
    q_rev = GaussianDistribution(mu=mu_rev, sigma=sigma_rev)
    q_rev_vals = torch.exp(q_rev.log_prob(x))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Forward KL
    ax1.plot(x, p_vals.detach().numpy(), 'k-', linewidth=2, label='P(x): Target')
    ax1.plot(x, q_fwd_vals.detach().numpy(), 'b-', linewidth=2, label=f'Q(x): N({mu_fwd:.2f}, {sigma_fwd:.2f}²)')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Forward KL: Mode-Covering', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    
    # Reverse KL
    ax2.plot(x, p_vals.detach().numpy(), 'k-', linewidth=2, label='P(x): Target')
    ax2.plot(x, q_rev_vals.detach().numpy(), 'r-', linewidth=2, label=f'Q(x): N({mu_rev:.2f}, {sigma_rev:.2f}²)')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('Reverse KL: Mode-Seeking', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kl_comparison.png', dpi=150)
    print("\n✓ Comparison saved as 'kl_comparison.png'")
    plt.show()

if __name__ == "__main__":
    p = GaussianMixtureDistribution(means=[-2, 6], sigmas=[1, 1], weights=[0.5, 0.5])
    
    mu_final, sigma_final, loss_hist = optimize_forward_kl(
        p=p,
        mu_init=0.0,
        sigma_init=1.0,
        lr=0.1,
        n_steps=1000,
        n_samples=1000
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"Final parameters: mu = {mu_final:.4f}, sigma = {sigma_final:.4f}")

    q_init = GaussianDistribution(mu=0.0, sigma=1.0)
    q_final = GaussianDistribution(mu=mu_final, sigma=sigma_final)
    plot_before_after(p, q_init, q_final, title="Forward KL Optimization")

    
    print("\n" + "=" * 50)
    print("Step 7: Reverse KL Optimization Test")
    print("=" * 50)
    
    # Initialize closer to one mode to break symmetry
    mu_final_rev, sigma_final_rev, loss_hist_rev = optimize_reverse_kl(
        p=p,
        mu_init=0.0,
        sigma_init=1.0,
        lr=0.01,
        n_steps=1000,
        n_samples=1000
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"Final parameters: mu = {mu_final_rev:.4f}, sigma = {sigma_final_rev:.4f}")

    q_init = GaussianDistribution(mu=0.0, sigma=1.0)
    q_final_rev = GaussianDistribution(mu=mu_final_rev, sigma=sigma_final_rev)
    plot_before_after(p, q_init, q_final_rev, title="Reverse KL Optimization")


    print("\n" + "=" * 50)
    print("Comparison: Forward KL vs Reverse KL")
    print("=" * 50)
    print(f"\nForward KL result:  mu = {mu_final:.4f}, sigma = {sigma_final:.4f} (mode-covering)")
    print(f"Reverse KL result:  mu = {mu_final_rev:.4f}, sigma = {sigma_final_rev:.4f} (mode-seeking)")
    
    # Comparison visualization
    compare_forward_reverse(p, mu_final, sigma_final, mu_final_rev, sigma_final_rev)


