from dataclasses import dataclass
import numpy as np

SEED = 1983
FEATURES = ['batting_average', 'strike_rate', 'boundary_pct', 'away_average']

@dataclass
class DataLayer:
    data_version: str = 'innings@2024.06'

    def __post_init__(self):
        rng = np.random.default_rng(SEED)
        self.X = rng.uniform([10, 60, 0.1, 8], [65, 160, 0.75, 60], size=(600, 4))
        self.group = rng.integers(0, 2, 600)
        self.y = ((0.03*self.X[:,0] + 0.02*self.X[:,1] + 1.5*self.X[:,2]) > 4.5).astype(int)

    def training_set(self):
        return self.X, self.y

    def reference_features(self):
        return self.X[:, 0]
