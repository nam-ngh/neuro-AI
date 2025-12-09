import torch
import torch.nn as nn

class RNNVanilla(nn.Module):
    def __init__(self, input_size, hidden_size, **kwargs):
        '''
        Vanilla RNN implementation
        
        :params input_size: size of input
        :params hidden_size: size of hidden state
        :params kwargs: absorbs extra arguments for compatibility
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
    
    def recur(self, X, h):
        hidden_new = h @ self.Whh.T
        input_term = X @ self.Wxh.T
        return torch.tanh(input_term + hidden_new + self.bh)
    
    def forward(self, X, h=None):
        steps, batch_size, _ = X.shape
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        
        outputs = []
        for t in range(steps):
            h = self.recur(X[t], h)
            outputs.append(h)
        
        out = torch.stack(outputs, dim=0)
        return out, h