import torch
import torch.nn as nn

class RNNBio(nn.Module):
    def __init__(self, input_size, hidden_size, r_excitatory=0.8, **kwargs):
        '''
        Biologically-inspired implementation of the RNN using Dale's Law, 
        modelling excitatory vs inhibitory pre-synaptic neurons 
        as columns of the recurrent weight matrix
        
        :params input_size: size of input
        :params hidden_size: size of hidden state
        :params r_excitatory: ratio of excitatory neurons / all neurons (0-1, default 0.8)
        :params kwargs:
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        # creating the excitatory-inhibitory (ei) mask
        ei_vec = torch.ones(hidden_size)
        ei_vec[int(hidden_size * r_excitatory):] = -1
        # ensure mask is accessible from device (GPU)
        self.register_buffer('ei_mask', torch.diag(ei_vec))

    def recur(self, X, h):
        hidden_new = h @ self.Whh.T
        # apply the ei mask
        masked_hidden_new = torch.relu(hidden_new) @ self.ei_mask
        input_term = X @ self.Wxh.T
        return torch.tanh(input_term + masked_hidden_new + self.bh)

    def forward(self, X, h=None):
        steps, batch_size, _ = X.shape
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=X.device)
        
        outputs = []
        for t in range(steps):
            h = self.recur(X[t], h)
            outputs.append(h)
        # stack outputs
        out = torch.stack(outputs, dim=0)
        return out, h