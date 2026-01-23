import numpy as np
import torch
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)


# n = 100000
# mu1, var1 = -5, 1
# mu2, var2 = 5, 1
# x = np.hstack((np.random.normal(mu1, var1, int(n/2)), np.random.normal(mu2, var2, int(n/2))))
# counts, bins = np.histogram(x, bins = 100, density=True)
# bins_centers = 0.5 * (bins[1:] + bins[:-1])

# plt.plot(bins_centers, counts, linewidth=2)
# plt.show()

def sample_from_reference_distribution(n_samples):
    """
    Mimics sampling from a real distribution. Similar to drawin text examples when training
    a language model
    """
    return np.random.choice([-5,5], n_samples) + np.random.normal(0, 1, n_samples)


samples = sample_from_reference_distribution(n_samples=100000)
counts, bins = np.histogram(samples, bins=50, density=True)
bins_centers = 0.5 * (bins[1:] + bins[:-1])
plt.figure()
# plt.hist(samples, bins=50, density=True, alpha=0.7, color='steelblue')
plt.plot(bins_centers, counts, linewidth=2)
plt.title('reference distribution P(x): Two Guassian Modes')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

class ApproxDistribution:
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def sample(self, n_samples):
        """Draw Samples from Q(x)"""
        return np.random.normal(self.mu, self.sigma, n_samples)
    
    def log_prob(self, x):
        """Compute log Q(x)"""
        return -0.5 * np.log(2*np.pi*self.sigma**2) - 0.5 * (x - self.mu)**2 / (self.sigma**2)
    
def empirical_expectation_under_P(f, n_samples=10000):
    """
    Estimate E_{x ~ P}[f(x)] using Monte Carlo Sampling
    """
    x_samples = sample_from_reference_distribution(n_samples)
    values = f(x_samples)
    return np.mean(values)

def estimate_forward_KL(model, n_samples=10000):
    """
    Monte Carlo Estimation of E_P[-log Q(x)]
    """
    x = sample_from_reference_distribution(n_samples)
    return empirical_expectation_under_P(lambda x: -model.log_prob(x))
    # log_q = model.log_prob(x)
    # return -np.mean(log_q)

def log_p_reference(x):
    """
    Log-density of the reference distribution P(x),
    a 50-50 mixture of N(-5, 1) and N(5, 1).
    logp(x)=log(exp(logp1(x))+exp(logp2(x))).
    We use log-sum-exp for numerical stability.
    log(ea+eb)=max(a,b)+log(ea-max(a,b)+eb-max(a,b)).
    """
    def log_gaussian(x, mu, sigma):
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * (x - mu)**2 / (sigma**2)

    log_p1 = np.log(0.5) + log_gaussian(x, -5, 1)
    log_p2 = np.log(0.5) + log_gaussian(x, 5, 1)
    max_log = np.maximum(log_p1, log_p2)
    log_p = max_log + np.log(np.exp(log_p1 - max_log) + np.exp(log_p2 - max_log))
    return log_p


def estimate_reverse_KL(model, n_samples=10000):
    x = model.sample(n_samples)
    log_q = model.log_prob(x)
    log_p = log_p_reference(x)
    return np.mean(log_q - log_p)





