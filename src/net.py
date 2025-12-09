import torch.nn as nn

class Net(nn.Module):
    def __init__(self, rnn_class, input_size, hidden_size, output_size, **kwargs):
        '''
        General wrapper class for rnn classes

        :param rnn_class: RNNBio or RNNVanilla
        :param input_size: size of input
        :param hidden_size: size of hidden state
        :param output_size: size of output
        '''

        super().__init__()
        self.rnn = rnn_class(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)
        print(f'Building {rnn_class} model with hidden size {hidden_size}')
        print(f'Input size: {input_size}, Output size: {output_size} ')
    
    def forward(self, x):
        rnn_output, _ = self.rnn(x)
        out = self.fc(rnn_output)
        return out, rnn_output