import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import wandb
from torch_geometric.nn import GATConv # Retorna ao GATConv
from torch_geometric.data import Data, Batch
from magent2.environments.battle_v4 import parallel_env
from copy import deepcopy 

# -------------------
# Configurações (AJUSTADAS)
# -------------------

MAP_SIZE = 45
N_AGENT = 81    # Mantemos 20
N_ENEMY = 12
ACTION_DIM = 9
GAMMA = 0.99
LR = 1e-5       
BATCH_SIZE = 64
TAU = 0.001
EPISODES = 100000
MAX_STEPS = 300
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.9995 # Retorna ao decaimento mais lento para estabilidade
REAL_INPUT_DIM = 5
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Curriculum params
K_START = 1
K_MAX = 4
K_GROW_EP = 5000   

# Enemy curriculum thresholds
ENEMY_PHASE_1 = 1000   
ENEMY_PHASE_2 = 5000   

# -------------------
# Inicialização WandB (Mock para compatibilidade e logs)
# -------------------
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

# **OPÇÃO**: Use o MockWandB se não quiser logar, caso contrário, use o wandb.init real.
# wandb = MockWandB() 

if 'wandb' not in globals() or not isinstance(wandb, MockWandB):
    wandb.init(
        project="dgn-magent-improved",
        config={
            "map_size": MAP_SIZE, "n_agent": N_AGENT, "n_enemy": N_ENEMY, "action_dim": ACTION_DIM,
            "gamma": GAMMA, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU, "eps_start": EPS_START,
            "eps_end": EPS_END, "eps_decay": EPS_DECAY, "real_input_dim": REAL_INPUT_DIM, "hidden_dim": HIDDEN_DIM,
        },
        reinit=True
    )
config = wandb.config


# -------------------
# Utils (Mantidas do DGN original, pois são mais robustas)
# -------------------
def normalize_obs(x):
    x = x.clone()
    if x.numel() == 0: return x
    if x[:, 0].max() > 0: x[:, 0] = x[:, 0] / 10.0 
    if float(MAP_SIZE) > 0:
        x[:, -2] = x[:, -2] / float(MAP_SIZE)
        x[:, -1] = x[:, -1] / float(MAP_SIZE)
    alive_mask = (x.abs().sum(dim=1) > 1e-6) 
    x[~alive_mask] = 0.0 
    return x


def observation_torch_fixed(obs_dict, agent_list):
    """
    Constrói o estado (N_AGENT, REAL_INPUT_DIM) preservando a observação
    do MAgent sem destruir a estrutura da grade.
    Extrai:
        - mapa local 13x13x5 achatado
        - HP
        - coordenadas do agente
    """
    N = len(agent_list)
    
    # shapes fixos do battle_v4
    OBS_H, OBS_W, OBS_C = 13, 13, 5
    GRID_SIZE = OBS_H * OBS_W * OBS_C

    REAL_INPUT_DIM = GRID_SIZE + 3   # mapa + HP + (x, y)

    state = torch.zeros((N, REAL_INPUT_DIM), dtype=torch.float32)
    alive_mask = torch.zeros(N, dtype=torch.bool)

    for i, name in enumerate(agent_list):

        if name not in obs_dict:
            continue  # agente morto

        obs = np.asarray(obs_dict[name])

        # A observação deve ser algo como (13, 13, 5)
        if obs.shape != (OBS_H, OBS_W, OBS_C):
            print(f"[WARN] Shape inesperado de {name}: {obs.shape}")
            continue

        # ------- 1) Flatten do mapa 13x13x5 --------
        flat_map = obs.reshape(-1)  # 845 valores

        # ------- 2) HP (canal 0 centro do grid) -----
        hp = obs[OBS_H // 2, OBS_W // 2, 0]

        # ------- 3) Posição (canal 1 e 2) -----------
        pos_x = obs[OBS_H // 2, OBS_W // 2, 1]
        pos_y = obs[OBS_H // 2, OBS_W // 2, 2]

        # ------- Montar vetor final -----------------
        full_vec = np.concatenate([flat_map, [hp, pos_x, pos_y]])

        state[i] = torch.tensor(full_vec, dtype=torch.float32)
        alive_mask[i] = True

    return state, alive_mask



def build_knn_edge_index_fast(x, k=4):
    """ (Mantida a lógica de KNN baseada em coordenadas) """
    if x.numel() == 0: return torch.empty((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.bool)
    coords = x[:, -2:]
    alive_mask_graph = (x.abs().sum(dim=1) > 1e-6)
    alive_idx_global = alive_mask_graph.nonzero(as_tuple=False).squeeze(1) 
    if alive_idx_global.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long), alive_mask_graph

    coords_alive = torch.nan_to_num(coords[alive_idx_global], nan=0.0)
    dist = torch.cdist(coords_alive, coords_alive, p=2)
    N_alive = coords_alive.size(0)

    k_eff = min(k + 1, N_alive)
    vals, idx = dist.topk(k=k_eff, largest=False)
    
    rows, cols = [], []
    for i in range(N_alive):
        neighs_local = [n for n in idx[i].tolist() if n != i][:k]
        idx_global_source = alive_idx_global[i].item()
        for n_local in neighs_local:
            idx_global_target = alive_idx_global[n_local].item()
            rows.append(idx_global_source)
            cols.append(idx_global_target)
            
    if len(rows) == 0: return torch.empty((2, 0), dtype=torch.long), alive_mask_graph
        
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index, alive_mask_graph


def get_batch_graph(states_tensor, k=4):
    """ (Mantida a lógica de PyG Batch/Mapeamento) """
    data_list = []
    for s in states_tensor: 
        edge_index_full, alive_mask_graph = build_knn_edge_index_fast(s, k=k)
        alive_idx_global = alive_mask_graph.nonzero(as_tuple=False).squeeze(1)
        
        if alive_idx_global.numel() == 0:
            x, edge_index = torch.zeros((0, REAL_INPUT_DIM), dtype=torch.float32), torch.empty((2, 0), dtype=torch.long)
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

# -------------------
# Modelos (Retornados ao GAT original)
# -------------------
REAL_INPUT_DIM = 848   # 13*13*5 + 3

class FeatureMLP(nn.Module):
    def __init__(self, input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)

        self.fc2 = nn.Linear(256, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if x.numel() == 0:
            return x

        x = F.relu(self.ln1(self.fc1(x)))
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
        if feat.numel() == 0: return torch.empty(0, ACTION_DIM, device=feat.device) 
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

# -------------------
# Replay Buffer (Mantido)
# -------------------
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

# -------------------
# Inicialização Ambiente e Agentes (CORREÇÃO DA EXTRAÇÃO DE NOMES)
# -------------------

env = parallel_env(map_size=MAP_SIZE, max_cycles=MAX_STEPS, render_mode="none")


agent = DGNAgent().to(DEVICE)
target_agent = DGNAgent().to(DEVICE)
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=LR)
buffer = ReplayBuffer()
eps = EPS_START

# Obter a lista de nomes dos agentes (ordem FIXA) e dos inimigos
obs_reset = env.reset()
all_possible_agents = sorted(env.possible_agents)

# Garantir que agent_list e enemy_list tenham 20 e 12 agentes, respectivamente
if len(all_possible_agents) > 0:
    # Agrupamento heurístico: o primeiro prefixo é o time do agente, o último é o inimigo
    # Ex: 'agent_0', 'agent_19', 'enemy_0', 'enemy_11'
    agent_handle_prefix = all_possible_agents[0].rsplit('_', 1)[0]
    enemy_handle_prefix = all_possible_agents[-1].rsplit('_', 1)[0]
    
    agent_list = sorted([a for a in env.possible_agents if a.startswith(agent_handle_prefix)])
    enemy_list = sorted([a for a in env.possible_agents if a.startswith(enemy_handle_prefix)])
else:
    # Fallback se o ambiente não retornar agentes
    agent_list = [f'agent_{i}' for i in range(N_AGENT)]
    enemy_list = [f'enemy_{i}' for i in range(N_ENEMY)]


if len(agent_list) != N_AGENT:
    print(f"AVISO: N_AGENT definido ({N_AGENT}) não bate com ambiente ({len(agent_list)}). Ajustando N_AGENT.")
    N_AGENT = len(agent_list)
    
# Helper: soft update
def soft_update(local_model, target_model, tau):
    for param, target_param in zip(local_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# Helper: enemy policy curriculum (Mantida)
def enemy_action_policy(ep, enemy_obs, agent_positions_list):
    if ep < ENEMY_PHASE_1: return 0  
    if ep < ENEMY_PHASE_2: return np.random.randint(0, ACTION_DIM)
    
    try:
        coords = np.asarray(enemy_obs)
        if coords.ndim > 1:
            coords = coords.reshape(coords.shape[0], -1)
            coords = coords[:, -2:]
            
            if coords.shape[0] == 0: return np.random.randint(0, ACTION_DIM)
            
            e = np.random.randint(0, coords.shape[0])
            ex, ey = coords[e]
            
            if len(agent_positions_list) == 0: return np.random.randint(0, ACTION_DIM)
            ap = np.array(agent_positions_list)
            
            d = np.sum((ap - np.array([ex, ey]))**2, axis=1)
            idx = np.argmin(d)
            dx, dy = ap[idx] - np.array([ex, ey])
            
            ax = 1 if dx > 0.5 else (-1 if dx < -0.5 else 0)
            ay = 1 if dy > 0.5 else (-1 if dy < -0.5 else 0)
            
            if ax == 0 and ay == 0: return 0
            if ax == 0 and ay == 1: return 1
            if ax == 0 and ay == -1: return 2
            if ax == -1 and ay == 0: return 3
            if ax == 1 and ay == 0: return 4
            if ax == -1 and ay == 1: return 5
            if ax == 1 and ay == 1: return 6
            if ax == -1 and ay == -1: return 7
            if ax == 1 and ay == -1: return 8
    except Exception:
        pass
    return np.random.randint(0, ACTION_DIM)
    """
Execute este script ANTES do treinamento para diagnosticar o problema.
"""
import numpy as np
from magent2.environments.battle_v4 import parallel_env

MAP_SIZE = 45
MAX_STEPS = 300

env = parallel_env(map_size=MAP_SIZE, max_cycles=MAX_STEPS, render_mode="none")
obs = env.reset()

print("\n" + "="*80)
print("DIAGNÓSTICO DO AMBIENTE MAGENT2")
print("="*80)

# 1. Agentes disponíveis
print(f"\n1. AGENTES NO AMBIENTE:")
print(f"   Total: {len(env.possible_agents)}")
print(f"   Lista: {env.possible_agents[:5]}... (mostrando 5)")

# 2. Separar agentes e inimigos
all_agents = sorted(env.possible_agents)
agent_prefix = all_agents[0].rsplit('_', 1)[0]
enemy_prefix = all_agents[-1].rsplit('_', 1)[0]

agents = [a for a in all_agents if a.startswith(agent_prefix)]
enemies = [a for a in all_agents if a.startswith(enemy_prefix)]

print(f"\n2. DIVISÃO:")
print(f"   Agentes ({agent_prefix}): {len(agents)}")
print(f"   Inimigos ({enemy_prefix}): {len(enemies)}")

# 3. Estrutura da observação
first_agent = agents[0]
if first_agent in obs:
    sample_obs = obs[first_agent]
    print(f"\n3. OBSERVAÇÃO DO AGENTE '{first_agent}':")
    print(f"   Tipo: {type(sample_obs)}")
    print(f"   Shape: {sample_obs.shape if hasattr(sample_obs, 'shape') else len(sample_obs)}")
    print(f"   DType: {sample_obs.dtype if hasattr(sample_obs, 'dtype') else 'N/A'}")
    print(f"   Min: {sample_obs.min():.2f}, Max: {sample_obs.max():.2f}")
    print(f"   Primeiros 20 valores:")
    print(f"   {sample_obs[:20]}")
else:
    print(f"\n[ERRO] Agente '{first_agent}' não encontrado nas observações!")

# 4. Executar um step
actions = {agent: 0 for agent in agents}  # Ação 0 = não mover
actions.update({enemy: 0 for enemy in enemies})

next_obs, rewards, terminated, truncated, info = env.step(actions)

print(f"\n4. APÓS 1 STEP:")
print(f"   Agentes vivos: {sum(1 for a in agents if a in next_obs)}/{len(agents)}")
print(f"   Reward total: {sum(rewards.values()):.2f}")
print(f"   Rewards individuais (primeiros 5):")
for i, agent_name in enumerate(agents[:5]):
    r = rewards.get(agent_name, 0.0)
    print(f"      {agent_name}: {r:.2f}")

# 5. Executar 10 steps aleatórios
print(f"\n5. TESTE DE 10 STEPS ALEATÓRIOS:")
env.reset()
total_r = 0
for step in range(10):
    actions = {agent: np.random.randint(0, 9) for agent in agents}
    actions.update({enemy: 0 for enemy in enemies})
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    step_reward = sum(rewards.values())
    total_r += step_reward
    
    alive_count = sum(1 for a in agents if a in obs)
    print(f"   Step {step+1}: reward={step_reward:.2f}, vivos={alive_count}/{len(agents)}")
    
    if terminated.get(agents[0], False):
        print(f"   [TERMINOU no step {step+1}]")
        break

print(f"\n   Reward total dos 10 steps: {total_r:.2f}")

print("\n" + "="*80)
print("DIAGNÓSTICO COMPLETO")
print("="*80)
# -------------------
# Loop de Treino (Mantido o DGN/GAT original)
# -------------------
for ep in range(EPISODES):
    if ep == 0:
         obs = obs_reset 
    else:
         obs = env.reset()

    total_reward = 0.0
    k = K_START + int((K_MAX - K_START) * min(1.0, ep / K_GROW_EP))

    for step in range(MAX_STEPS):
        # OBTENÇÃO DA OBSERVAÇÃO (Dicionário de PettingZoo)
        obs_agents_dict = {name: obs[name] for name in agent_list if name in obs}
        obs_enemies_dict = {name: obs[name] for name in enemy_list if name in obs}

        # PASSO 1: Processar estado atual (na ordem FIXA)
        state, alive_mask_original = observation_torch_fixed(obs_agents_dict, agent_list)
        state_normalized = normalize_obs(state)
        
        # PASSO 2: Construir grafo
        edge_index, alive_mask_graph = build_knn_edge_index_fast(state_normalized, k=k)
        alive_idx_global = alive_mask_graph.nonzero(as_tuple=False).squeeze(1)
        num_agents_alive = alive_idx_global.numel()

        # Seleção de Ação (Epsilon-Greedy)
        action_alive = torch.zeros(num_agents_alive, dtype=torch.long)
        
        if num_agents_alive > 0:
            agent.eval()
            with torch.no_grad():
                x_nodes = state_normalized[alive_idx_global].to(DEVICE)
                ei = edge_index.to(DEVICE)
                q_nodes = agent(x_nodes, ei) 
            
            for i in range(num_agents_alive):
                if random.random() < eps:
                    action_alive[i] = random.randint(0, ACTION_DIM - 1)
                else:
                    action_alive[i] = torch.argmax(q_nodes[i].cpu())

        # PASSO 3: Mapear ações para o dicionário do ambiente
        actions_dict = {}
        action_buffer = torch.zeros(N_AGENT, dtype=torch.long) 

        for i, global_idx in enumerate(alive_idx_global.tolist()):
            agent_name = agent_list[global_idx]
            action = int(action_alive[i].item())
            actions_dict[agent_name] = action
            action_buffer[global_idx] = action 

        # Ações do inimigo pelo curriculum
        agent_positions_list = state[alive_mask_original][:, -2:].numpy().tolist() 
        for name, obs_e in obs_enemies_dict.items():
            action = enemy_action_policy(ep, obs_e, agent_positions_list)
            actions_dict[name] = int(action)

        # Passo no ambiente
        next_obs, reward, terminated, truncated, info = env.step(actions_dict)
        done = terminated
        
        # Processar próximo estado
        next_obs_agents_dict = {name: next_obs[name] for name in agent_list if name in next_obs}
        next_state, next_alive_mask_original = observation_torch_fixed(next_obs_agents_dict, agent_list)
        next_state_normalized = normalize_obs(next_state)

        # Processar recompensas (na ordem FIXA)
        rewards_tensor = torch.zeros(N_AGENT, dtype=torch.float32)
        for i, name in enumerate(agent_list):
            rewards_tensor[i] = reward.get(name, 0.0) 

        agent_terminal = done.get(agent_list[0], False) or truncated.get(agent_list[0], False)

        # Adicionar ao buffer (agora na ordem FIXA)
        buffer.add(state_normalized, action_buffer, next_state_normalized, rewards_tensor, agent_terminal)

        # Decaimento do epsilon
        eps = max(EPS_END, eps * EPS_DECAY)
        obs = next_obs
        total_reward += sum(reward.values()) if reward else 0

        if agent_terminal:
            break

        # -------------------
        # Treinamento (GNN Batch)
        # -------------------
        if len(buffer) > BATCH_SIZE:
            batch_samples = buffer.sample(BATCH_SIZE)
            states = torch.stack([b[0] for b in batch_samples])           
            actions = torch.stack([b[1] for b in batch_samples])          
            next_states = torch.stack([b[2] for b in batch_samples])     
            rewards = torch.stack([b[3] for b in batch_samples])         
            dones = torch.tensor([b[4] for b in batch_samples], dtype=torch.float32) 

            batch_graph = get_batch_graph(states, k=k)
            batch_graph_next = get_batch_graph(next_states, k=k)

            x, edge_index = batch_graph.x.to(DEVICE), batch_graph.edge_index.to(DEVICE) if batch_graph.edge_index.numel()>0 else torch.empty((2,0), dtype=torch.long, device=DEVICE)
            x_next, edge_index_next = batch_graph_next.x.to(DEVICE), batch_graph_next.edge_index.to(DEVICE) if batch_graph_next.edge_index.numel()>0 else torch.empty((2,0), dtype=torch.long, device=DEVICE)

            agent.train()
            q_all = agent(x, edge_index) 

            with torch.no_grad():
                q_next_all_target = target_agent(x_next, edge_index_next)

            
            # --- Mapeamento das Q-Values para o loss (Double DQN) ---
            q_pred_list = []
            q_target_list = []
            
            ptr = batch_graph.ptr.cpu().numpy()
            ptr_next = batch_graph_next.ptr.cpu().numpy()

            for b_i in range(states.size(0)):
                alive_mask_b = batch_graph.alive_mask[b_i]
                alive_idx_global = alive_mask_b.nonzero(as_tuple=False).squeeze(1)
                num_alive = alive_idx_global.numel()
                
                if num_alive == 0:
                    continue

                # 1. Q_PRED (Online Network)
                start_n = int(ptr[b_i])
                end_n = int(ptr[b_i+1])
                q_nodes = q_all[start_n:end_n] 

                chosen_actions_global = actions[b_i][alive_idx_global].to(DEVICE) 
                q_pred_vals = q_nodes.gather(1, chosen_actions_global.unsqueeze(1)).squeeze(1) 

                # 2. Q_TARGET (Double DQN)
                rewards_alive = rewards[b_i][alive_idx_global].to(DEVICE)
                done_flag = dones[b_i].to(DEVICE)
                
                start_next = int(ptr_next[b_i])
                end_next = int(ptr_next[b_i+1])
                
                q_next_nodes_online = agent(x_next[start_next:end_next], edge_index_next)
                best_next_actions = q_next_nodes_online.max(dim=1)[1]

                q_next_nodes_target = q_next_all_target[start_next:end_next]
                q_next_target_vals = q_next_nodes_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

                q_target_vals = rewards_alive + GAMMA * q_next_target_vals * (1.0 - done_flag)

                q_pred_list.append(q_pred_vals)
                q_target_list.append(q_target_vals)

            if len(q_pred_list) == 0:
                continue

            q_pred_all_concat = torch.cat(q_pred_list)
            q_target_all_concat = torch.cat(q_target_list)

            loss = F.mse_loss(q_pred_all_concat, q_target_all_concat)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            soft_update(agent, target_agent, TAU)

            wandb.log(**{"loss": loss.item(), "episode": ep, "q_mean": q_pred_all_concat.mean().item()})

    print(f"Episode {ep:05d}, total_reward={total_reward:.2f}, epsilon={eps:.4f}, k={k}")
    wandb.log({
        "total_reward": total_reward,
        "epsilon": eps,
        "k_neighbors": k,
        "episode": ep
    })


    if ep > 0 and ep % 100 == 0:
        torch.save(agent.state_dict(), "dgn_agent_improved.pth")
        wandb.save("dgn_agent_improved.pth")