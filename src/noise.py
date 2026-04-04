import numpy as np


def apply_label_noise(y, noise_rate=0.0, random_state=None):
    if noise_rate == 0.0:
        return y.copy()
    rng = np.random.default_rng(random_state)
    y_noisy = y.copy()
    n_flip = int(noise_rate * len(y))
    flip_indices = rng.choice(len(y), size=n_flip, replace=False)
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    return y_noisy


def apply_feature_noise(X, noise_std=0.0, random_state=None):
    if noise_std == 0.0:
        return X.copy()
    rng = np.random.default_rng(random_state)
    return X + rng.normal(0, noise_std, size=X.shape)


def apply_weak_samples(X, y, weak_fraction=0.0, noise_std=0.01, random_state=None):
    if weak_fraction == 0.0:
        return X.copy(), y.copy()
    rng = np.random.default_rng(random_state)
    n_weak = int(weak_fraction * X.shape[0])
    indices = rng.choice(X.shape[0], size=n_weak, replace=False)
    X_weak = X[indices] + rng.normal(0, noise_std, size=X[indices].shape)
    return np.vstack([X, X_weak]), np.concatenate([y, y[indices].copy()])