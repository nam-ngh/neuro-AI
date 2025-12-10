from src.rnn_bio import RNNBio
from src.rnn_vanilla import RNNVanilla

DT = 20
DEVICE = 'cuda'

RNN_VANILLA = {
    'name': 'vanilla',
    'rnn_class': RNNVanilla,
    'hidden_size': 128,
}
RNN_BIO = {
    'name': 'bio',
    'rnn_class': RNNBio,
    'hidden_size': 128,
    'r_excitatory': 0.8,
}
DMS_TRAINING_CONFIGS = {
    'vanilla': {
        'seq_len': 480,
        'epochs': 200,
        'lr': 0.001,
    },
    'bio': {
        'seq_len': 480,
        'l1_lambda': 0.001,
        'epochs': 200,
        'lr': 0.001,
    }
}
GNG_TRAINING_CONFIGS = {
    'vanilla': {
        'seq_len': 225,
        'epochs': 200,
        'lr': 0.001,
    },
    'bio': {
        'seq_len': 225,
        'l1_lambda': 0.001,
        'epochs': 200,
        'lr': 0.001,
    }
}