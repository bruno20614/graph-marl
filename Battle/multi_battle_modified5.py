import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from magent2.environments.battle_v4 import parallel_env
import wandb

MAP_SIZE = 45
ACTION_DIM = 21
GAMMA = 0.99
LR = 5e-4
BATCH_SIZE = 32
TAU = 0.01
EPISODES = 3000
MAX_STEPS = 160
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.99995
REAL_INPUT_DIM = 845 + 2  # 13*13*5 + coordenadas (x,y)
HIDDEN_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_EVERY = 8

# KNN config
K_START = 4
K_MAX = 4
K_GROW_EP = 5000
KNN_RADIUS = 0.25

print(f"\n{'='*80}")
print(f"DGN-MAGENT - VERSÃO CORRIGIDA")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
print(f"Input Dim: {REAL_INPUT_DIM} (obs + coords) | Actions: {ACTION_DIM}")
print(f"Grafo: KNN Espacial com raio {KNN_RADIUS}")
print(f"{'='*80}\n")

wandb.init(
    project="DGN-MAGENT",
    name="dgn_shaped_knn",
    config={
        "map_size": MAP_SIZE,
        "lr": LR,
        "gamma": GAMMA,
        "hidden_dim": HIDDEN_DIM,
        "max_steps": MAX_STEPS
    }
)


config = wandb.config

# ==============================================================================
# EXTRAÇÃO DE POSIÇÕES E OBSERVAÇÕES
# ==============================================================================

def extract_positions_from_obs(obs_dict, agent_list, map_size=MAP_SIZE):
    """
    Extrai posições (x, y) dos agentes a partir do canal de observação.
    O agente está sempre no centro do grid 13x13, então usamos isso como referência.
    """
    positions = {}
    
    for name in agent_list:
        if name not in obs_dict:
            continue
            
        obs = np.asarray(obs_dict[name], dtype=np.float32)
        if obs.shape != (13, 13, 5):
            continue
        
        agent_idx = agent_list.index(name)
        x = (agent_idx % int(np.sqrt(len(agent_list)))) * (map_size / np.sqrt(len(agent_list)))
        y = (agent_idx // int(np.sqrt(len(agent_list)))) * (map_size / np.sqrt(len(agent_list)))
        
        # Adicionar ruído baseado na observação para diferenciar
        own_channel = obs[:, :, 0]
        x_offset = np.argmax(own_channel.sum(axis=0)) - 6  # Centro em 6
        y_offset = np.argmax(own_channel.sum(axis=1)) - 6
        
        x = np.clip(x + x_offset * 2, 0, map_size - 1)
        y = np.clip(y + y_offset * 2, 0, map_size - 1)
        
        positions[name] = (x, y)
    
    return positions


def observation_to_tensor(obs_dict, agent_list, positions):
    """
    Converte observações em tensor COM coordenadas espaciais.
    """
    N = len(agent_list)
    OBS_H, OBS_W, OBS_C = 13, 13, 5
    GRID_SIZE = OBS_H * OBS_W * OBS_C
    
    state = torch.zeros((N, GRID_SIZE + 2), dtype=torch.float32)  # +2 para (x,y)
    alive_mask = torch.zeros(N, dtype=torch.bool)
    enemies_visible = torch.zeros(N, dtype=torch.float32)
    coords = torch.zeros((N, 2), dtype=torch.float32)
    
    for i, name in enumerate(agent_list):
        if name not in obs_dict:
            continue
        
        obs = np.asarray(obs_dict[name], dtype=np.float32)
        
        if obs.shape != (OBS_H, OBS_W, OBS_C):
            continue
        
        # Flatten do grid
        flat_obs = obs.reshape(-1)
        
        # Adicionar coordenadas
        if name in positions:
            x, y = positions[name]
            # Normalizar coordenadas para [0, 1]
            x_norm = x / MAP_SIZE
            y_norm = y / MAP_SIZE
            coords[i] = torch.tensor([x_norm, y_norm], dtype=torch.float32)
            
            # Concatenar observação + coordenadas
            state[i] = torch.cat([
                torch.tensor(flat_obs, dtype=torch.float32),
                coords[i]
            ])
            alive_mask[i] = True
            
            # Contar inimigos visíveis (canal 1)
            enemy_channel = obs[:, :, 1]
            enemies_visible[i] = (enemy_channel > 0).sum()
    
    return state, alive_mask, enemies_visible, coords


# ==============================================================================
# REWARD SHAPING CORRIGIDO
# ==============================================================================

def compute_shaped_reward(base_rewards, enemies_visible_before, enemies_visible_after, 
                          actions, coords, alive_mask):
    
    shaped_rewards = torch.zeros_like(base_rewards)
    
    # 1. Bônus por aumentar visibilidade (PESO AUMENTADO)
    visibility_increase = enemies_visible_after - enemies_visible_before
    shaped_rewards += visibility_increase * 0.5  # Era 0.01
    
    # 2. Bônus por atacar com inimigos visíveis (PESO AUMENTADO)
    is_attack_action = (actions >= 9).float()
    has_enemies = (enemies_visible_before > 0).float()
    attack_bonus = is_attack_action * has_enemies * 1.0  # Era 0.05
    shaped_rewards += attack_bonus
    
    # 3. Penalidade por ficar parado (PESO AUMENTADO)
    idle_penalty = (actions == 0).float() * -0.5  # Era -0.02
    shaped_rewards += idle_penalty
    
    # 4. Bônus por formação (agentes próximos uns dos outros)
    if alive_mask.sum() > 1:
        alive_coords = coords[alive_mask]
        if alive_coords.shape[0] > 1:
            # Calcular distância média entre agentes vivos
            dists = torch.cdist(alive_coords, alive_coords)
            avg_dist = dists[dists > 0].mean()
            # Bônus se distância média for moderada (não muito longe, não muito perto)
            formation_bonus = torch.exp(-((avg_dist - 0.3) ** 2) / 0.1) * 0.2
            shaped_rewards[alive_mask] += formation_bonus
    
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
# GRAFO KNN ESPACIAL (CORRIGIDO)
# ==============================================================================

def build_spatial_knn_graph(coords, alive_mask, k=4, radius=KNN_RADIUS):

    if coords.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.bool)
    
    alive_idx = alive_mask.nonzero(as_tuple=False).squeeze(1)
    
    if alive_idx.numel() <= 1:
        return torch.empty((2, 0), dtype=torch.long), alive_mask
    
    # Coordenadas dos agentes vivos
    alive_coords = coords[alive_idx]  # (N_alive, 2)
    
    # Calcular distâncias euclidianas entre todos os pares
    dists = torch.cdist(alive_coords, alive_coords)  # (N_alive, N_alive)
    
    # Construir arestas
    rows, cols = [], []
    
    for i in range(alive_idx.numel()):
        # Pegar distâncias do agente i para todos os outros
        dist_i = dists[i].clone()
        dist_i[i] = float('inf')  # Ignorar si mesmo
        
        # Aplicar filtro de raio
        valid_neighbors = (dist_i <= radius)

        
        if valid_neighbors.sum() == 0:
            continue
        
        # Pegar k vizinhos mais próximos dentro do raio
        dist_i[~valid_neighbors] = float('inf')
        _, nearest_k = torch.topk(dist_i, min(k, valid_neighbors.sum()), largest=False)
        
        src_global = alive_idx[i].item()
        for neighbor_local in nearest_k:
            if dist_i[neighbor_local] < float('inf'):
                tgt_global = alive_idx[neighbor_local].item()
                rows.append(src_global)
                cols.append(tgt_global)
    
    if len(rows) == 0:
        return torch.empty((2, 0), dtype=torch.long), alive_mask
    
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index, alive_mask


def get_batch_graph(states_tensor, coords_tensor, k=4):
    """
    Cria batch de grafos COM remapeamento correto de índices.
    """
    data_list = []
    
    for s, c in zip(states_tensor, coords_tensor):
        # Máscara de agentes vivos
        alive_mask = (s.abs().sum(dim=1) > 1e-6)
        
        # Construir grafo espacial
        edge_index_full, _ = build_spatial_knn_graph(c, alive_mask, k=k)
        
        alive_idx_global = alive_mask.nonzero(as_tuple=False).squeeze(1)
        
        if alive_idx_global.numel() == 0:
            x = torch.zeros((0, REAL_INPUT_DIM), dtype=torch.float32)
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            x = s[alive_idx_global]
            
            # Remapear edge_index de global para local
            global_to_local = {g.item(): l for l, g in enumerate(alive_idx_global)}
            
            if edge_index_full.numel() > 0:
                sources = edge_index_full[0].tolist()
                targets = edge_index_full[1].tolist()
                
                rows_local = []
                cols_local = []
                for src, tgt in zip(sources, targets):
                    if src in global_to_local and tgt in global_to_local:
                        rows_local.append(global_to_local[src])
                        cols_local.append(global_to_local[tgt])
                
                if len(rows_local) > 0:
                    edge_index = torch.tensor([rows_local, cols_local], dtype=torch.long)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index)
        data.alive_mask = alive_mask
        data_list.append(data)
    
    return Batch.from_data_list(data_list)


# ===================== EXPLORAÇÃO GUIADA =====================
def guided_exploration_action(ep, q_values):
    if ep < 100:
        return random.randint(0, ACTION_DIM - 1)
    else:
        return torch.argmax(q_values).item()


# ==============================================================================
# MODELOS (IGUAIS)
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
    def __init__(self, hidden_dim=HIDDEN_DIM, heads=2):
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

    def add(self, s, a, s_next, r, done, coords, coords_next):
        item = (s.clone(), a.clone(), s_next.clone(), r.clone(), done, 
                coords.clone(), coords_next.clone())
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
# LOOP DE TREINO (CORRIGIDO)
# ==============================================================================

for ep in range(EPISODES):
    obs = obs_reset if ep == 0 else env.reset()
    
    total_reward = 0.0
    shaped_reward_sum = 0.0
    #k = K_START + int((K_MAX - K_START) * min(1.0, ep / K_GROW_EP))
    k = 4
    for step in range(MAX_STEPS):
        obs_agents_dict = {name: obs[name] for name in agent_list if name in obs}
        obs_enemies_dict = {name: obs[name] for name in enemy_list if name in obs}
        
        # Extrair posições espaciais
        positions = extract_positions_from_obs(obs_agents_dict, agent_list)
        
        state, alive_mask_original, enemies_visible_before, coords = \
            observation_to_tensor(obs_agents_dict, agent_list, positions)
        
        state_normalized = normalize_obs(state)
        
        # Construir grafo KNN espacial
        edge_index, alive_mask_graph = build_spatial_knn_graph(coords, alive_mask_original, k=k)
        alive_idx_global = alive_mask_graph.nonzero(as_tuple=False).squeeze(1)
        num_agents_alive = alive_idx_global.numel()
        
        action_alive = torch.zeros(num_agents_alive, dtype=torch.long)
        
        if num_agents_alive > 0:
            agent.eval()
            with torch.no_grad():
                x_nodes = state_normalized[alive_idx_global].to(DEVICE)
                
                # Remapear edge_index
                if edge_index.numel() > 0:
                    global_to_local = {g.item(): l for l, g in enumerate(alive_idx_global)}
                    src_local = [global_to_local.get(s.item(), -1) for s in edge_index[0]]
                    tgt_local = [global_to_local.get(t.item(), -1) for t in edge_index[1]]
                    
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
                if ep < 100:
                    # exploração total inicial
                    action_alive[i] = random.randint(0, ACTION_DIM - 1)
                else:
                    # ε-greedy normal
                    if random.random() < eps:
                        action_alive[i] = random.randint(0, ACTION_DIM - 1)
                    else:
                        action_alive[i] = torch.argmax(q_nodes[i]).item()

        
        # --- BLOCO DE AÇÕES CORRIGIDO ---
        actions_dict = {}
        action_buffer = torch.zeros(N_AGENT, dtype=torch.long)
        
        # Pegamos o conjunto de agentes que o ambiente REALMENTE espera agora
        current_env_agents = set(env.agents)

        # 1. Preenche ações para seus agentes (Blue) que estão vivos no GNN e no Env
        for i, global_idx in enumerate(alive_idx_global.tolist()):
            agent_name = agent_list[global_idx]
            
            # Só adiciona se o agente ainda estiver na lista oficial do MAgent2
            if agent_name in current_env_agents:
                action = int(action_alive[i].item())
                actions_dict[agent_name] = action
                action_buffer[global_idx] = action

        # 2. Preenche ações para os inimigos (Red) que estão vivos
        for name in enemy_list:
            if name in current_env_agents:
                # É mais seguro usar o action_space do que np.random.randint
                actions_dict[name] = env.action_space(name).sample()

        # 3. VERIFICAÇÃO CRÍTICA: Se não há ações (todos morreram), interrompe o passo
        if not actions_dict:
            break

        # Agora o step é seguro
        next_obs, reward, terminated, truncated, info = env.step(actions_dict)
        
        # Simplifica o estado de 'terminado'
        done = terminated or truncated
        

        # Próximo estado
        next_positions = extract_positions_from_obs(
            {name: next_obs[name] for name in agent_list if name in next_obs}, 
            agent_list
        )
        next_state, _, enemies_visible_after, coords_next = \
            observation_to_tensor(
                {name: next_obs[name] for name in agent_list if name in next_obs},
                agent_list, next_positions
            )
        next_state_normalized = normalize_obs(next_state)
        
        # Rewards
        rewards_tensor = torch.zeros(N_AGENT, dtype=torch.float32)
        for i, name in enumerate(agent_list):
            rewards_tensor[i] = reward.get(name, 0.0)
        
        # Reward Shaping CORRIGIDO
        shaped_rewards = compute_shaped_reward(
            rewards_tensor, 
            enemies_visible_before, 
            enemies_visible_after, 
            action_buffer,
            coords,
            alive_mask_original
        )

        # ===== CORREÇÃO FUNDAMENTAL =====
        shaping_weight = max(0.05, 0.3 - ep / 3000)
        alive_count = alive_mask_original.sum().clamp(min=1)
        total_rewards = rewards_tensor + shaping_weight * (shaped_rewards / alive_count)

        agent_terminal = len([a for a in agent_list if a in next_obs]) == 0

        
        buffer.add(
            state_normalized,
            action_buffer,
            next_state_normalized, 
            total_rewards,  
            agent_terminal,
            coords,
            coords_next
        )

        
        if ep >= 100:
            eps = max(EPS_END, eps * EPS_DECAY)

        obs = next_obs
        total_reward += sum(reward.get(a, 0.0) for a in agent_list)
        shaped_reward_sum += (shaping_weight * (shaped_rewards / alive_count)).sum().item()

        
        if agent_terminal:
            break
        
        # ===== TREINAMENTO DDQN VETORIZADO (GPU FRIENDLY) =====
        if step % TRAIN_EVERY == 0 and len(buffer) > BATCH_SIZE:
            batch_samples = buffer.sample(BATCH_SIZE)
            
            states = torch.stack([b[0] for b in batch_samples]).to(DEVICE)
            actions = torch.stack([b[1] for b in batch_samples]).to(DEVICE)
            next_states = torch.stack([b[2] for b in batch_samples]).to(DEVICE)
            rewards = torch.stack([b[3] for b in batch_samples]).to(DEVICE)
            dones = torch.tensor([b[4] for b in batch_samples], dtype=torch.float32).to(DEVICE)
            coords_batch = torch.stack([b[5] for b in batch_samples])
            coords_next_batch = torch.stack([b[6] for b in batch_samples])
            
            batch_graph = get_batch_graph(states.cpu(), coords_batch, k=k).to(DEVICE)
            batch_graph_next = get_batch_graph(next_states.cpu(), coords_next_batch, k=k).to(DEVICE)
            
            agent.train()
            
            q_values_all = agent(batch_graph.x, batch_graph.edge_index)
            
            flat_actions = actions[batch_graph.alive_mask.view(BATCH_SIZE, N_AGENT)]
            q_pred = q_values_all.gather(1, flat_actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                q_next_online = agent(batch_graph_next.x, batch_graph_next.edge_index)
                next_actions = q_next_online.argmax(dim=1, keepdim=True)
                
                q_next_target_all = target_agent(batch_graph_next.x, batch_graph_next.edge_index)
                q_next_max_all = q_next_target_all.gather(1, next_actions).squeeze(1)

            mask_t = batch_graph.alive_mask.view(-1)
            mask_t_plus_1 = batch_graph_next.alive_mask.view(-1)
            survived_mask = mask_t_plus_1[mask_t]
            
            q_next_aligned = torch.zeros(q_pred.size(0), device=DEVICE)
            q_next_aligned[survived_mask] = q_next_max_all
            
            flat_rewards = rewards.view(-1)[mask_t]
            nodes_per_graph = torch.bincount(batch_graph.batch)
            flat_dones = torch.repeat_interleave(dones, nodes_per_graph)
            
            q_target = flat_rewards + (GAMMA * q_next_aligned * (1.0 - flat_dones))
            
            loss = F.mse_loss(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
            optimizer.step()
            
            soft_update(agent, target_agent, TAU)

            
            wandb.log({"loss": loss.item(), "episode": ep, "q_mean": q_pred.mean().item()})
    
    print(f"Ep {ep:05d} | Base R: {total_reward:7.2f} | Shaped R: {shaped_reward_sum:7.2f} | ε: {eps:.4f} | k: {k}")
    wandb.log({
        "total_reward": total_reward, 
        "shaped_reward": shaped_reward_sum, 
        "epsilon": eps, 
        "k_neighbors": k, 
        "episode": ep
    })
    
    if ep > 0 and ep % 100 == 0:
        torch.save(agent.state_dict(), "dgn_shaped.pth")
        print(f"[SAVE] ep {ep}")

print("\n" + "="*80)
print("CONCLUÍDO!")
print("="*80)