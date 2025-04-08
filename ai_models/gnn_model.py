import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class HardwareTrojanGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim)
        self.conv2 = pyg_nn.GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = pyg_nn.global_mean_pool(x, data.batch)
        return self.classifier(x)

class GNNTrainer:
    def __init__(self, model, device='cuda', learning_rate=0.001):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, loader):
        """Train the model for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = self.criterion(out, data.y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.y.size(0)
            
        accuracy = correct / total if total > 0 else 0
        return total_loss / len(loader)
    
    def validate(self, loader):
        """Validate the model on a validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                loss = self.criterion(out, data.y)
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.y.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return total_loss / len(loader)
    
    def predict(self, data):
        """Make predictions on new data"""
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            out = self.model(data)
            pred = out.argmax(dim=1)
            confidence = torch.softmax(out, dim=1)
            
        return pred, confidence
