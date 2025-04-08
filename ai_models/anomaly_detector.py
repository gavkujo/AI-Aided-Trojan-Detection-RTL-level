import numpy as np
import torch
from torch import nn, optim
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class RTLAutoencoder(nn.Module):
    """Unsupervised anomaly detector for RTL features"""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetector:
    def __init__(self, method='autoencoder', config=None):
        self.method = method
        self.scaler = StandardScaler()
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, features):
        """Train detector on normal samples"""
        if self.method == 'iforest':
            contamination = self.config.get('contamination', 0.01)
            self.model = IsolationForest(contamination=contamination)
            self.model.fit(self.scaler.fit_transform(features))
        elif self.method == 'autoencoder':
            latent_dim = self.config.get('latent_dim', 32)
            self.model = RTLAutoencoder(features.shape[1], latent_dim).to(self.device)
            self._train_autoencoder(features)

    def _train_autoencoder(self, features):
        """Train autoencoder model on normal data"""
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        tensor_features = torch.FloatTensor(scaled_features).to(self.device)
        
        # Training parameters
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(tensor_features, tensor_features)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_features, _ in dataloader:
                # Forward pass
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_features)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
                
        self.model.eval()

    def detect(self, features):
        """Detect anomalous samples"""
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        if self.method == 'iforest':
            # IsolationForest returns 1 for inliers and -1 for outliers
            # Convert to anomaly scores (higher = more anomalous)
            predictions = self.model.predict(scaled_features)
            return np.where(predictions == -1, 1.0, 0.0)
        else:
            # For autoencoder, compute reconstruction error as anomaly score
            tensor_features = torch.FloatTensor(scaled_features).to(self.device)
            with torch.no_grad():
                reconstructed = self.model(tensor_features)
                mse = torch.mean((reconstructed - tensor_features)**2, dim=1)
            return mse.cpu().numpy()
