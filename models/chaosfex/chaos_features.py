"""
ChaosFEX: Chaos-based Feature Extraction for Retinal Disease Classification

Implements chaos-based feature extraction using:
1. Generalized Luroth Series (GLS) map
2. Logistic map

Extracts four key features per chaotic neuron:
- Mean Firing Time (MFT)
- Mean Firing Rate (MFR)  
- Mean Energy (ME)
- Mean Entropy (MEnt)

Reference: ChaosFEX-NGRC-RRDC project integration
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Tuple, Optional, List
from scipy.stats import entropy


def GLS_map(x: float, b: float) -> float:
    """
    Generalized Luroth Series (GLS) map
    
    x_{n+1} = x_n + b * x_n^2 (mod 1)
    
    Args:
        x: Current state (0 < x < 1)
        b: GLS parameter
        
    Returns:
        Next state
    """
    return (x + b * x**2) % 1.0


def logistic_map(x: float, r: float) -> float:
    """
    Logistic map for chaotic dynamics
    
    x_{n+1} = r * x_n * (1 - x_n)
    
    Args:
        x: Current state (0 < x < 1)
        r: Logistic parameter (3.6 < r < 4.0 for chaos)
        
    Returns:
        Next state
    """
    return r * x * (1 - x)


class ChaosFEXExtractor:
    """
    Chaos-based Feature Extractor
    
    Transforms deep learning features through chaotic dynamics to extract
    robust, nonlinear features for classification.
    
    Args:
        n_neurons: Number of chaotic neurons (default: 100)
        map_type: Type of chaotic map ('GLS', 'Logistic', 'Hybrid')
        b: Parameter for GLS map (default: 0.1)
        r: Parameter for Logistic map (default: 3.8)
        threshold: Firing threshold (default: 0.5)
        max_iterations: Maximum iterations for chaotic dynamics (default: 1000)
    """
    
    def __init__(
        self,
        n_neurons: int = 100,
        map_type: Literal['GLS', 'Logistic', 'Hybrid'] = 'GLS',
        b: float = 0.1,
        r: float = 3.8,
        threshold: float = 0.5,
        max_iterations: int = 1000,
        random_seed: Optional[int] = None
    ):
        self.n_neurons = n_neurons
        self.map_type = map_type
        self.b = b
        self.r = r
        self.threshold = threshold
        self.max_iterations = max_iterations
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def compute_firing_times(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute firing times when trajectory crosses threshold"""
        crossings = np.where(np.diff((trajectory > self.threshold).astype(int)) == 1)[0]
        return crossings
    
    def compute_features(self, trajectory: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute ChaosFEX features from trajectory
        
        Args:
            trajectory: Chaotic trajectory
            
        Returns:
            Tuple of (MFT, MFR, ME, MEnt)
        """
        # Mean Firing Time (MFT)
        firing_times = self.compute_firing_times(trajectory)
        if len(firing_times) > 1:
            mft = np.mean(np.diff(firing_times))
        else:
            mft = self.max_iterations
        
        # Mean Firing Rate (MFR)
        mfr = len(firing_times) / self.max_iterations
        
        # Mean Energy (ME)
        me = np.mean(trajectory**2)
        
        # Mean Entropy (MEnt)
        hist, _ = np.histogram(trajectory, bins=20, range=(0, 1), density=True)
        hist = hist / (np.sum(hist) + 1e-10)
        ment = entropy(hist + 1e-10)
        
        return mft, mfr, me, ment
    
    def generate_trajectory(self, initial_condition: float) -> np.ndarray:
        """Generate chaotic trajectory from initial condition"""
        trajectory = np.zeros(self.max_iterations)
        trajectory[0] = initial_condition
        
        for i in range(1, self.max_iterations):
            if self.map_type == 'GLS':
                trajectory[i] = GLS_map(trajectory[i-1], self.b)
            elif self.map_type == 'Logistic':
                trajectory[i] = logistic_map(trajectory[i-1], self.r)
            else:  # Hybrid
                if i % 2 == 0:
                    trajectory[i] = GLS_map(trajectory[i-1], self.b)
                else:
                    trajectory[i] = logistic_map(trajectory[i-1], self.r)
        
        return trajectory
    
    def map_input_to_initial_conditions(self, input_vector: np.ndarray) -> np.ndarray:
        """Map input feature vector to initial conditions for chaotic neurons"""
        # Normalize to [0, 1]
        input_min = input_vector.min()
        input_max = input_vector.max()
        input_norm = (input_vector - input_min) / (input_max - input_min + 1e-10)
        
        if len(input_norm) < self.n_neurons:
            # Repeat and add noise for diversity
            repeats = int(np.ceil(self.n_neurons / len(input_norm)))
            input_norm = np.tile(input_norm, repeats)[:self.n_neurons]
            input_norm += np.random.uniform(0, 0.01, self.n_neurons)
        else:
            # Sample uniformly
            indices = np.linspace(0, len(input_norm)-1, self.n_neurons, dtype=int)
            input_norm = input_norm[indices]
        
        return np.clip(input_norm, 0.01, 0.99)
    
    def extract_features(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Extract ChaosFEX features from input vector
        
        Args:
            input_vector: Input feature vector (D-dimensional)
            
        Returns:
            ChaosFEX features (4 * n_neurons dimensional)
        """
        initial_conditions = self.map_input_to_initial_conditions(input_vector)
        
        features = []
        for ic in initial_conditions:
            trajectory = self.generate_trajectory(ic)
            mft, mfr, me, ment = self.compute_features(trajectory)
            features.extend([mft, mfr, me, ment])
        
        return np.array(features)
    
    def extract_features_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """Extract ChaosFEX features from batch of input vectors"""
        batch_features = []
        for input_vector in input_batch:
            features = self.extract_features(input_vector)
            batch_features.append(features)
        return np.array(batch_features)
    
    def get_output_dim(self) -> int:
        """Get output dimension of ChaosFEX features"""
        return 4 * self.n_neurons
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability"""
        feature_names = []
        for i in range(self.n_neurons):
            feature_names.extend([
                f'Neuron{i}_MFT',
                f'Neuron{i}_MFR',
                f'Neuron{i}_ME',
                f'Neuron{i}_MEnt'
            ])
        return feature_names


class MultiScaleChaosFEX:
    """
    Multi-scale ChaosFEX extractor with different parameter settings
    Captures dynamics at multiple temporal scales for richer representations
    """
    
    def __init__(
        self,
        n_neurons_per_scale: int = 50,
        n_scales: int = 3,
        map_type: Literal['GLS', 'Logistic'] = 'GLS'
    ):
        self.n_scales = n_scales
        self.n_neurons_per_scale = n_neurons_per_scale
        self.extractors = []
        
        if map_type == 'GLS':
            b_values = np.linspace(0.1, 0.3, n_scales)
            for b in b_values:
                self.extractors.append(
                    ChaosFEXExtractor(n_neurons=n_neurons_per_scale, map_type='GLS', b=b)
                )
        else:
            r_values = np.linspace(3.6, 3.95, n_scales)
            for r in r_values:
                self.extractors.append(
                    ChaosFEXExtractor(n_neurons=n_neurons_per_scale, map_type='Logistic', r=r)
                )
    
    def extract_features(self, input_vector: np.ndarray) -> np.ndarray:
        """Extract multi-scale ChaosFEX features"""
        all_features = []
        for extractor in self.extractors:
            features = extractor.extract_features(input_vector)
            all_features.append(features)
        return np.concatenate(all_features)
    
    def extract_features_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """Extract multi-scale features from batch"""
        batch_features = []
        for input_vector in input_batch:
            features = self.extract_features(input_vector)
            batch_features.append(features)
        return np.array(batch_features)
    
    def get_output_dim(self) -> int:
        """Get total output dimension"""
        return 4 * self.n_neurons_per_scale * self.n_scales


class ChaosFEXLayer(nn.Module):
    """
    PyTorch-compatible ChaosFEX layer for end-to-end training
    
    This wraps the ChaosFEX extractor to work within PyTorch models,
    though the chaotic dynamics themselves are computed in NumPy.
    """
    
    def __init__(
        self,
        n_neurons: int = 100,
        map_type: str = 'GLS',
        b: float = 0.1,
        max_iterations: int = 500
    ):
        super().__init__()
        self.extractor = ChaosFEXExtractor(
            n_neurons=n_neurons,
            map_type=map_type,
            b=b,
            max_iterations=max_iterations
        )
        self.output_dim = self.extractor.get_output_dim()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ChaosFEX layer
        
        Args:
            x: Input tensor (batch_size, feature_dim)
            
        Returns:
            ChaosFEX features (batch_size, 4*n_neurons)
        """
        device = x.device
        x_np = x.detach().cpu().numpy()
        
        chaos_features = self.extractor.extract_features_batch(x_np)
        
        return torch.from_numpy(chaos_features).float().to(device)


if __name__ == "__main__":
    print("ChaosFEX Feature Extractor Test")
    print("=" * 50)
    
    # Basic test
    extractor = ChaosFEXExtractor(n_neurons=10, map_type='GLS')
    input_vec = np.random.randn(512)
    features = extractor.extract_features(input_vec)
    
    print(f"Input dimension: {len(input_vec)}")
    print(f"Output dimension: {len(features)}")
    print(f"Feature sample: {features[:8]}")
    
    # Multi-scale test
    ms_extractor = MultiScaleChaosFEX(n_neurons_per_scale=5, n_scales=3)
    ms_features = ms_extractor.extract_features(input_vec)
    print(f"\nMulti-scale output dim: {len(ms_features)}")
    
    # PyTorch layer test
    layer = ChaosFEXLayer(n_neurons=10)
    x = torch.randn(4, 512)
    out = layer(x)
    print(f"\nPyTorch layer output shape: {out.shape}")
