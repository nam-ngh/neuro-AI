import neurogym as ngym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List

from src.net import Net

class Experiment:
    def __init__(self, env_name, dt=20, device='cpu'):
        self.env_name = env_name
        self.device = device
        self.dt = dt
        
        # get env dimensions
        env = ngym.make(env_name, dt=dt)
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        
        # store results
        self.hidden_hist = {}
        self.results = {}
        self.models = {}
    
    def build_model(self, **rnn_kwargs):
        return Net(
            input_size=self.input_size,
            output_size=self.output_size,
            **rnn_kwargs
        ).to(self.device)
    
    def train(
            self,
            model,
            name,
            epochs=100,
            batch_size=32,
            seq_len=100,
            lr=0.001,
            l1_lambda=None,
            clip_grad=1.0,
            print_every=10,
            save_hidden_every=10,
    ):
        env = ngym.make(self.env_name, dt=self.dt)
        dataset = ngym.Dataset(env, batch_size=batch_size, seq_len=seq_len)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        accuracies = []
        hidden_hist = {}
        
        for epoch in range(epochs):
            model.train()
            
            inputs, labels = dataset()
            inputs = torch.from_numpy(inputs).float().to(self.device)
            labels = torch.from_numpy(labels).long().to(self.device)
            
            optimizer.zero_grad()
            outputs, hiddens = model(inputs)
            
            task_loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            
            if l1_lambda is not None:
                # impose sparsity if needed
                l1_loss = l1_lambda * torch.sum(torch.abs(model.rnn.Whh))
                loss = task_loss + l1_loss
            else:
                loss = task_loss
            
            loss.backward()
            if clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            
            with torch.no_grad():
                # get prediction
                pred = outputs.argmax(dim=-1)
                # select only non-0 labels sequence for comparison
                mask = labels != 0
                acc = (pred[mask] == labels[mask]).float().mean().item() if mask.sum() > 0 else 0.0
            
            losses.append(loss.item())
            accuracies.append(acc)
            if epoch % save_hidden_every == 0:
                hidden_hist[epoch] = hiddens[:, 0, :].detach().cpu().numpy()
            
            if epoch % print_every == 0:
                print(f"[{name}] Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.4f}")
        
        # store last hidden states
        hidden_hist[epoch -1] = hiddens[:, 0, :].detach().cpu().numpy()
        # store results
        self.hidden_hist[name] = hidden_hist
        self.results[name] = {'losses': losses, 'accuracies': accuracies}
        self.models[name] = model
        
        return self
    
    def compare(self, models_configs: List[Dict], train_configs: Dict):
        """
        configs: dict of {name: {'rnn_class': class, **rnn_kwargs}}
        """
        for model_dict in models_configs:
            name = model_dict['name']
            print(f"\nTRAINING >>>>>>>>>>>>> {name}\n")
            model = self.build_model(**model_dict)
            tr_config = train_configs[name]
            self.train(model, name, **tr_config)
        
        return self
    
    def plot(self,):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        for name, result in self.results.items():
            axes[0].plot(result['losses'], label=name)
            axes[1].plot(result['accuracies'], label=name)
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def summary(self):
        print("EXPERIMENT SUMMARY")
        for name, result in self.results.items():
            final_loss = result['losses'][-1]
            final_acc = result['accuracies'][-1]
            best_acc = max(result['accuracies'])
            print(f"{name}: Final Loss={final_loss:.4f}, Final Acc={final_acc:.4f}, Best Acc={best_acc:.4f}")
        
        return self