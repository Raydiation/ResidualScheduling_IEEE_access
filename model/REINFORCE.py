import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch.distributions import Categorical
from itertools import accumulate
from model.gnn import GNN
torch.set_printoptions(precision=10)

class REINFORCE(nn.Module):
    def __init__(self, args):
        super(REINFORCE, self).__init__()
        self.args = args
        self.num_layers = args.policy_num_layers
        self.hidden_dim = args.hidden_dim
        self.gnn = GNN(args)
        self.layers = torch.nn.ModuleList()
        self.layers.append(nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim))
        for _ in range(self.num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, 1))
        
        self.log_probs = []
        self.entropies = []
        self.rewards = []
        self.baselines = []
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, avai_ops, data, op_unfinished, max_process_time, greedy=False):
        x_dict = self.gnn(data)

        score = torch.empty(size=(0, self.args.hidden_dim * 2 + 1)).to(self.args.device)

        for op_info in avai_ops:
            score = torch.cat((score, torch.cat((x_dict['m'][op_info['m_id']],
                                                x_dict['op'][op_unfinished.index(op_info['node_id'])],
                                                torch.tensor(op_info['process_time'] / max_process_time).to(torch.float32).to(self.args.device).unsqueeze(0)), dim=0).unsqueeze(0)), dim=0)

        for i in range(self.num_layers - 1):
            score = F.leaky_relu(self.layers[i](score))
        score = self.layers[self.num_layers - 1](score)

        probs = F.softmax(score, dim=0).flatten()
        dist = Categorical(probs)
        if greedy == True:
            idx = torch.argmax(score)
        else:
            idx = dist.sample()
        self.log_probs.append(dist.log_prob(idx))
        self.entropies.append(dist.entropy())
        return idx.item(), probs[idx].item()
    
    def calculate_loss(self, device):
        loss = []
        returns = torch.FloatTensor(list(accumulate(self.rewards[::-1]))[::-1]).to(device)
        policy_loss = 0.0
        entropy_loss = 0.0

        for log_prob, entropy, R, baseline in zip(self.log_probs, self.entropies, returns, self.baselines):
            if baseline == 0:
                advantage = R * -1
            else:
                advantage = ((R - baseline) / baseline) * -1
            loss.append(-log_prob * advantage - self.args.entropy_coef * entropy)
            policy_loss += log_prob * advantage
            entropy_loss += entropy
        
        return torch.stack(loss).mean(), policy_loss / len(self.log_probs), entropy_loss / len(self.log_probs)
 
    def clear_memory(self):
        del self.log_probs[:]
        del self.entropies[:]
        del self.rewards[:]
        del self.baselines[:]
        return
         