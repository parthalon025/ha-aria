"""Autoencoder using MLPRegressor for reconstruction-based anomaly feature extraction."""

import os
import pickle

HAS_SKLEARN = True
try:
    import numpy as np
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    HAS_SKLEARN = False


class Autoencoder:
    """Autoencoder built on MLPRegressor — trains to reconstruct its input.

    Reconstruction error (per-sample MSE) serves as an additional feature
    for downstream anomaly detectors like IsolationForest.
    """

    def __init__(self, hidden_layers=(24, 12, 24), max_iter=200):
        self.hidden_layers = hidden_layers
        self.max_iter = max_iter

    def train(self, X, model_dir):
        """Train the autoencoder on input data X.

        Fits a StandardScaler + MLPRegressor(X -> X) and saves both to model_dir.

        Args:
            X: array-like of shape (n_samples, n_features)
            model_dir: directory to save autoencoder.pkl and ae_scaler.pkl

        Returns:
            dict with training metrics or error.
        """
        if not HAS_SKLEARN:
            return {"error": "sklearn not installed"}

        X_arr = np.array(X, dtype=float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        model.fit(X_scaled, X_scaled)

        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "autoencoder.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, "ae_scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

        return {"samples": len(X_arr), "hidden_layers": list(self.hidden_layers)}

    def reconstruction_errors(self, X, model_dir):
        """Compute per-sample MSE reconstruction errors.

        Args:
            X: array-like of shape (n_samples, n_features)
            model_dir: directory containing autoencoder.pkl and ae_scaler.pkl

        Returns:
            numpy array of shape (n_samples,) with reconstruction errors,
            or None if model files are missing or sklearn unavailable.
        """
        if not HAS_SKLEARN:
            return None

        ae_path = os.path.join(model_dir, "autoencoder.pkl")
        scaler_path = os.path.join(model_dir, "ae_scaler.pkl")
        if not os.path.isfile(ae_path) or not os.path.isfile(scaler_path):
            return None

        with open(ae_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        X_arr = np.array(X, dtype=float)
        X_scaled = scaler.transform(X_arr)
        X_reconstructed = model.predict(X_scaled)

        # Per-sample MSE
        errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
        return errors
