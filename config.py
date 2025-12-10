from src.rnn_bio import RNNBio
from src.rnn_vanilla import RNNVanilla

DT = 20
DEVICE = 'cuda'

RNN_VANILLA = {
    'name': 'vanilla',
    'rnn_class': RNNVanilla,
    'hidden_size': 32,
}
RNN_BIO = {
    'name': 'bio',
    'rnn_class': RNNBio,
    'hidden_size': 32,
    'r_excitatory': 0.8,
}
DMS_TRAINING_CONFIGS = {
    'vanilla': {
        'seq_len': 160,
        'epochs': 150,
        'lr': 0.01,
    },
    'bio': {
        'seq_len': 160,
        'l1_lambda': 0.00001,
        'epochs': 150,
        'lr': 0.01,
    }
}
DC_TRAINING_CONFIGS = {
    'vanilla': {
        'seq_len': 260,
        'epochs': 220,
        'lr': 0.01,
    },
    'bio': {
        'seq_len': 260,
        'l1_lambda': 0.00001,
        'epochs': 220,
        'lr': 0.01,
    }
}