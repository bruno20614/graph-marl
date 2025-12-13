import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from magent2.environments.battle_v4 import parallel_env

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

MAP_SIZE = 45
ACTION_DIM = 21  # MAgent2 battle_v4 tem 21 ações (9 movimento + 12 ataque)
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 64
TAU = 0.01
EPISODES = 100000
MAX_STEPS = 300
EPS_START = 1.0
EPS_END = 0.5  # Mantém mais exploração
EPS_DECAY = 0.99995
REAL_INPUT_DIM = 845  # 13*13*5
HIDDEN_DIM = 256  # Aumentado
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

K_START = 4
K_MAX = 16
K_GROW_EP = 5000

print(f"\n{'='*80}")
print(f"DGN-MAGENT - VERSÃO COM REWARD SHAPING")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Input Dim: {REAL_INPUT_DIM} | Actions: {ACTION_DIM}")
print(f"{'='*80}\n")

# ==============================================================================
# MOCK WANDB
# ==============================================================================

class MockWandB:
    def __init__(self):
        self.config = {}
    def init(self, **kwargs):
        if 'config' in kwargs:
            self.config.update(kwargs['config'])
    def log(self, *args, **kwargs): 
        pass
    def save(self, *args): 
        pass

wandb = MockWandB()
config = wandb.config

# ==============================================================================
# OBSERVAÇÃO + REWARD SHAPING
# ==============================================================================

def observation_to_tensor(obs_dict, agent_list):
    """
    Converte observações em tensor com informação sobre inimigos visíveis.
    """
    N = len(agent_list)
    OBS_H, OBS_W, OBS_C = 13, 13, 5
    GRID_SIZE = OBS_H * OBS_W * OBS_C
    
    state = torch.zeros((N, GRID_SIZE), dtype=torch.float32)
    alive_mask = torch.zeros(N, dtype=torch.bool)
    enemies_visible = torch.zeros(N, dtype=torch.float32)  # Conta inimigos visíveis
    
    for i, name in enumerate(agent_list):
        if name not in obs_dict:
            continue
        
        obs = np.asarray(obs_dict[name], dtype=np.float32)
        
        if obs.shape != (OBS_H, OBS_W, OBS_C):
            continue
        
        # Flatten do grid
        flat_obs = obs.reshape(-1)
        state[i] = torch.tensor(flat_obs, dtype=torch.float32)
        alive_mask[i] = True
        
        # Contar inimigos visíveis (canal 1)
        enemy_channel = obs[:, :, 1]
        enemies_visible[i] = (enemy_channel > 0).sum()
    
    return state, alive_mask, enemies_visible


def compute_shaped_reward(base_rewards, enemies_visible_before, enemies_visible_after, actions):
    """
    Reward Shaping:
    - Bônus por VER mais inimigos (incentiva aproximação)
    - Bônus por ATACAR quando há inimigos visíveis
    - Penalidade por ficar parado
    """
    shaped_rewards = base_rewards.clone()
    
    # Bônus por aumentar visibilidade de inimigos
    visibility_increase = enemies_visible_after - enemies_visible_before
    shaped_rewards += visibility_increase * 0.01
    
    # Bônus por atacar quando há inimigos visíveis
    is_attack_action = (actions >= 9).float()  # Ações 9-20 são ataques
    has_enemies = (enemies_visible_before > 0).float()
    attack_bonus = is_attack_action * has_enemies * 0.05
    shaped_rewards += attack_bonus
    
    # Penalidade leve por ação 0 (não fazer nada)
    idle_penalty = (actions == 0).float() * -0.02
    shaped_rewards += idle_penalty
    
    return shaped_rewards


def normalize_obs(x):
    if x.numel() == 0:
        return x
    x = x.clone()
    x = torch.clamp(x, 0, 1)
    alive_mask = (x.abs().sum(dim=1) > 1e-6)
    x[~alive_mask] = 0.0
    return x


# ==============================================================================
# GRAFO
# ==============================================================================

def build_graph(x, k=4):
    """
    Grafo por proximidade de índice.
    Retorna edge_index em ÍNDICES GLOBAIS (para manter compatibilidade).
    """
    if x.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.bool)
    
    alive_mask = (x.abs().sum(dim=1) > 1e-6)
    alive_idx = alive_mask.nonzero(as_tuple=False).squeeze(1)
    
    if alive_idx.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long), alive_mask
    
    N_alive = alive_idx.numel()
    rows, cols = [], []
    
    for i in range(N_alive):
        src_global = alive_idx[i].item()
        
        # Conectar aos k/2 vizinhos anteriores e k/2 posteriores
        neighbors_global = []
        for offset in range(1, k + 1):
            # Vizinho à frente
            if i + offset < N_alive:
                neighbors_global.append(alive_idx[i + offset].item())
            # Vizinho atrás
            if i - offset >= 0:
                neighbors_global.append(alive_idx[i - offset].item())
        
        # Limitar a k vizinhos
        neighbors_global = neighbors_global[:k]
        
        for tgt_global in neighbors_global:
            rows.append(src_global)
            cols.append(tgt_global)
    
    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long), alive_mask
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index, alive_mask


def get_batch_graph(states_tensor, k=4):
    data_list = []
    for s in states_tensor:
        edge_index_full, alive_mask_graph = build_graph(s, k=k)
        alive_idx_global = alive_mask_graph.nonzero(as_tuple=False).squeeze(1)
        
        if alive_idx_global.numel() == 0:
            x = torch.zeros((0, REAL_INPUT_DIM), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            x = s[alive_idx_global]
            global_to_local = {g.item(): l for l, g in enumerate(alive_idx_global)}
            
            if edge_index_full.numel() > 0:
                sources, targets = edge_index_full[0].tolist(), edge_index_full[1].tolist()
                rows_local = [global_to_local[s] for s in sources]
                cols_local = [global_to_local[t] for t in targets]
                edge_index = torch.tensor([rows_local, cols_local], dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data.alive_mask = alive_mask_graph
        data_list.append(data)
    
    return Batch.from_data_list(data_list)


# ==============================================================================
# POLÍTICA DE EXPLORAÇÃO GUIADA
# ==============================================================================

def guided_exploration_action(ep, q_values):
    """
    Primeiros 100 episódios: Força movimento/ataque, não deixa ficar parado.
    Depois: epsilon-greedy normal.
    """
    if ep < 100:
        # Primeiros episódios: Sempre escolhe ação de movimento ou ataque
        # Exclui ação 0 (não fazer nada)
        if random.random() < 0.5:
            # 50%: Movimento (ações 1-8)
            return random.randint(1, 8)
        else:
            # 50%: Ataque (ações 9-20)
            return random.randint(9, ACTION_DIM - 1)
    else:
        # Depois de 100 eps: Usa a rede
        return torch.argmax(q_values).item()


# ==============================================================================
# MODELOS
# ==============================================================================

class FeatureMLP(nn.Module):
    def __init__(self, input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.numel() == 0:
            return x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        return x


class RelationGAT(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, heads=4):
        super().__init__()
        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, concat=True)

    def forward(self, x, edge_index):
        if x.numel() == 0 or edge_index.numel() == 0:
            return x
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        return x


class QNet(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 3, action_dim)

    def forward(self, feat, rel1, rel2):
        if feat.numel() == 0:
            return torch.empty(0, ACTION_DIM, device=feat.device)
        x = torch.cat([feat, rel1, rel2], dim=-1)
        return self.fc(x)


class DGNAgent(nn.Module):
    def __init__(self, input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.feature_mlp = FeatureMLP(input_dim, hidden_dim)
        self.relation1 = RelationGAT(hidden_dim)
        self.relation2 = RelationGAT(hidden_dim)
        self.q_net = QNet(hidden_dim, action_dim)

    def forward(self, x, edge_index):
        feat = self.feature_mlp(x)
        rel1 = self.relation1(feat, edge_index)
        rel2 = self.relation2(rel1, edge_index)
        q = self.q_net(feat, rel1, rel2)
        return q


# ==============================================================================
# REPLAY BUFFER
# ==============================================================================

class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, s, a, s_next, r, done):
        item = (s.clone(), a.clone(), s_next.clone(), r.clone(), done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        samples = random.sample(self.buffer, batch_size)
        return samples

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# INICIALIZAÇÃO
# ==============================================================================

env = parallel_env(map_size=MAP_SIZE, max_cycles=MAX_STEPS, render_mode="none")
obs_reset = env.reset()

all_possible_agents = sorted(env.possible_agents)
agent_list = sorted([a for a in all_possible_agents if a.startswith('blue')])
enemy_list = sorted([a for a in all_possible_agents if a.startswith('red')])

N_AGENT = len(agent_list)
N_ENEMY = len(enemy_list)

print(f"Agentes: {N_AGENT} | Inimigos: {N_ENEMY}\n")

agent = DGNAgent(input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
target_agent = DGNAgent(input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
target_agent.load_state_dict(agent.state_dict())

optimizer = optim.Adam(agent.parameters(), lr=LR)
buffer = ReplayBuffer()
eps = EPS_START

def soft_update(local_model, target_model, tau):
    for param, target_param in zip(local_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


# ==============================================================================
# LOOP DE TREINO
# ==============================================================================

for ep in range(EPISODES):
    obs = obs_reset if ep == 0 else env.reset()
    
    total_reward = 0.0
    shaped_reward_sum = 0.0
    k = K_START + int((K_MAX - K_START) * min(1.0, ep / K_GROW_EP))
    
    for step in range(MAX_STEPS):
        obs_agents_dict = {name: obs[name] for name in agent_list if name in obs}
        obs_enemies_dict = {name: obs[name] for name in enemy_list if name in obs}
        
        state, alive_mask_original, enemies_visible_before = observation_to_tensor(obs_agents_dict, agent_list)
        state_normalized = normalize_obs(state)
        
        edge_index, alive_mask_graph = build_graph(state_normalized, k=k)
        alive_idx_global = alive_mask_graph.nonzero(as_tuple=False).squeeze(1)
        num_agents_alive = alive_idx_global.numel()
        
        action_alive = torch.zeros(num_agents_alive, dtype=torch.long)
        
        if num_agents_alive > 0:
            agent.eval()
            with torch.no_grad():
                x_nodes = state_normalized[alive_idx_global].to(DEVICE)
                
                # Remapear edge_index de global para local
                if edge_index.numel() > 0:
                    global_to_local = {g.item(): l for l, g in enumerate(alive_idx_global)}
                    src_local = [global_to_local.get(s.item(), -1) for s in edge_index[0]]
                    tgt_local = [global_to_local.get(t.item(), -1) for t in edge_index[1]]
                    
                    # Filtrar arestas inválidas
                    valid_edges = [(s, t) for s, t in zip(src_local, tgt_local) if s >= 0 and t >= 0]
                    
                    if len(valid_edges) > 0:
                        src, tgt = zip(*valid_edges)
                        ei = torch.tensor([src, tgt], dtype=torch.long).to(DEVICE)
                    else:
                        ei = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
                else:
                    ei = torch.empty((2, 0), dtype=torch.long).to(DEVICE)
                
                q_nodes = agent(x_nodes, ei)
            
            for i in range(num_agents_alive):
                if random.random() < eps:
                    action_alive[i] = guided_exploration_action(ep, q_nodes[i].cpu())
                else:
                    action_alive[i] = torch.argmax(q_nodes[i].cpu())
        
        actions_dict = {}
        action_buffer = torch.zeros(N_AGENT, dtype=torch.long)
        
        for i, global_idx in enumerate(alive_idx_global.tolist()):
            agent_name = agent_list[global_idx]
            action = int(action_alive[i].item())
            actions_dict[agent_name] = action
            action_buffer[global_idx] = action
        
        # Inimigos aleatórios
        for name in enemy_list:
            if name in obs:
                actions_dict[name] = np.random.randint(0, ACTION_DIM)
        
        next_obs, reward, terminated, truncated, info = env.step(actions_dict)
        done = terminated
        
        next_obs_agents_dict = {name: next_obs[name] for name in agent_list if name in next_obs}
        next_state, _, enemies_visible_after = observation_to_tensor(next_obs_agents_dict, agent_list)
        next_state_normalized = normalize_obs(next_state)
        
        # Rewards base
        rewards_tensor = torch.zeros(N_AGENT, dtype=torch.float32)
        for i, name in enumerate(agent_list):
            rewards_tensor[i] = reward.get(name, 0.0)
        
        # Reward Shaping
        shaped_rewards = compute_shaped_reward(
            rewards_tensor, 
            enemies_visible_before, 
            enemies_visible_after, 
            action_buffer
        )
        
        agent_terminal = done.get(agent_list[0], False) or truncated.get(agent_list[0], False)
        
        buffer.add(state_normalized, action_buffer, next_state_normalized, shaped_rewards, agent_terminal)
        
        eps = max(EPS_END, eps * EPS_DECAY)
        obs = next_obs
        total_reward += sum(reward.values()) if reward else 0
        shaped_reward_sum += shaped_rewards.sum().item()
        
        if agent_terminal:
            break
        
        # Treinamento
        if len(buffer) > BATCH_SIZE:
            batch_samples = buffer.sample(BATCH_SIZE)
            states = torch.stack([b[0] for b in batch_samples])
            actions = torch.stack([b[1] for b in batch_samples])
            next_states = torch.stack([b[2] for b in batch_samples])
            rewards = torch.stack([b[3] for b in batch_samples])
            dones = torch.tensor([b[4] for b in batch_samples], dtype=torch.float32)
            
            batch_graph = get_batch_graph(states, k=k)
            batch_graph_next = get_batch_graph(next_states, k=k)
            
            x = batch_graph.x.to(DEVICE)
            edge_index = batch_graph.edge_index.to(DEVICE) if batch_graph.edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=DEVICE)
            x_next = batch_graph_next.x.to(DEVICE)
            edge_index_next = batch_graph_next.edge_index.to(DEVICE) if batch_graph_next.edge_index.numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=DEVICE)
            
            agent.train()
            q_all = agent(x, edge_index)
            
            with torch.no_grad():
                q_next_all_target = target_agent(x_next, edge_index_next)
            
            q_pred_list = []
            q_target_list = []
            
            ptr = batch_graph.ptr.cpu().numpy()
            ptr_next = batch_graph_next.ptr.cpu().numpy()
            
            for b_i in range(states.size(0)):
                alive_mask_b = batch_graph.alive_mask[b_i]
                alive_idx_b = alive_mask_b.nonzero(as_tuple=False).squeeze(1)
                num_alive = alive_idx_b.numel()
                
                if num_alive == 0:
                    continue
                
                start_n = int(ptr[b_i])
                end_n = int(ptr[b_i + 1])
                q_nodes = q_all[start_n:end_n]
                
                chosen_actions = actions[b_i][alive_idx_b].to(DEVICE)
                q_pred_vals = q_nodes.gather(1, chosen_actions.unsqueeze(1)).squeeze(1)
                
                rewards_alive = rewards[b_i][alive_idx_b].to(DEVICE)
                done_flag = dones[b_i].to(DEVICE)
                
                start_next = int(ptr_next[b_i])
                end_next = int(ptr_next[b_i + 1])
                
                q_next_online = agent(x_next[start_next:end_next], edge_index_next)
                best_next_actions = q_next_online.max(dim=1)[1]
                
                q_next_target = q_next_all_target[start_next:end_next]
                q_next_vals = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
                
                q_target_vals = rewards_alive + GAMMA * q_next_vals * (1.0 - done_flag)
                
                q_pred_list.append(q_pred_vals)
                q_target_list.append(q_target_vals)
            
            if len(q_pred_list) == 0:
                continue
            
            q_pred_all = torch.cat(q_pred_list)
            q_target_all = torch.cat(q_target_list)
            
            loss = F.mse_loss(q_pred_all, q_target_all)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
            optimizer.step()
            
            soft_update(agent, target_agent, TAU)
            
            wandb.log({"loss": loss.item(), "episode": ep, "q_mean": q_pred_all.mean().item()})
    
    print(f"Ep {ep:05d} | Base R: {total_reward:7.2f} | Shaped R: {shaped_reward_sum:7.2f} | ε: {eps:.4f} | k: {k}")
    wandb.log({"total_reward": total_reward, "shaped_reward": shaped_reward_sum, "epsilon": eps, "k_neighbors": k, "episode": ep})
    
    if ep > 0 and ep % 100 == 0:
        torch.save(agent.state_dict(), "dgn_shaped.pth")
        print(f"[SAVE] ep {ep}")

print("\n" + "="*80)
print("CONCLUÍDO!")
print("="*80)