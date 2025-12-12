import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import wandb
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
from magent2.environments.battle_v4 import parallel_env

# -------------------
# Configurações (AJUSTADAS)
# -------------------

MAP_SIZE = 45
N_AGENT = 20
N_ENEMY = 12
ACTION_DIM = 9
GAMMA = 0.99
LR = 1e-5        # AJUSTE CRÍTICO: Reduzido de 1e-4 para 1e-5 para combater loss=0.
BATCH_SIZE = 256
TAU = 0.01
EPISODES = 100000
MAX_STEPS = 300
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995 # AJUSTE CRÍTICO: Decaimento mais rápido (de 0.9999) para forçar explotação da rede Q treinada.
REAL_INPUT_DIM = 5
HIDDEN_DIM = 128

# -------------------
# Inicialização WandB
# -------------------
wandb.init(
    project="dgn-magent",
    config={
        "map_size": MAP_SIZE,
        "n_agent": N_AGENT,
        "n_enemy": N_ENEMY,
        "action_dim": ACTION_DIM,
        "gamma": GAMMA,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "tau": TAU,
        "eps_start": EPS_START,
        "eps_end": EPS_END,
        "eps_decay": EPS_DECAY,
        "real_input_dim": REAL_INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
    }
)
config = wandb.config

# -------------------
# Utils
# -------------------
def observation_torch(full_obs_agents, full_obs_enemies):
    """
    Processa a observação do MAgent para o formato de feature vetorial.
    Assume que o estado do agente (coord, HP, etc.) são as últimas 5 features
    do vetor de observação achatado (REAL_INPUT_DIM=5).
    """
    states = full_obs_agents if isinstance(full_obs_agents, list) else full_obs_agents
    states_np = np.asarray(states)
    
    if states_np.ndim > 2:
        num_agents_alive = states_np.shape[0]
        try:
            states_flat = states_np.reshape(num_agents_alive, -1)
            # Mantém apenas as últimas REAL_INPUT_DIM features (estado base do agente)
            if states_flat.shape[1] > REAL_INPUT_DIM:
                states_flat = states_flat[:, -REAL_INPUT_DIM:]
            return torch.tensor(states_flat, dtype=torch.float32)
        except ValueError:
            print(f"Alerta: Observação inconsistente. Shape recebido: {states_np.shape}")
            return torch.zeros(states_np.shape[0], REAL_INPUT_DIM, dtype=torch.float32)
    
    return torch.tensor(states, dtype=torch.float32)

def adjacency_graphsage(state, k=4):
    neighbors = []
    positions = []
    for j in range(len(state)):
        try:
            # As últimas duas features de state[j] são usadas como (X, Y)
            x_coord = np.array(state[j][-2]).flatten()
            y_coord = np.array(state[j][-1]).flatten()
            if x_coord.size > 0 and y_coord.size > 0:
                x, y = float(x_coord[0]), float(y_coord[0])
                positions.append((x, y))
            else:
                positions.append((-100.0, -100.0))
        except (IndexError, TypeError, AttributeError):
            positions.append((-100.0, -100.0))
    for j, (xj, yj) in enumerate(positions):
        if xj == -100.0:
            neighbors.append([])
            continue
        dist = []
        for i, (xi, yi) in enumerate(positions):
            if xi == -100.0 or i == j:
                continue
            dist.append((i, (xi - xj)**2 + (yi - yj)**2))
        dist.sort(key=lambda x: x[1])
        knn = [i for i, _ in dist[:k]]
        neighbors.append(knn)
    return neighbors

def neighbors_to_edge_index(neighbors):
    edge_index = []
    for i, neigh in enumerate(neighbors):
        for j in neigh:
            edge_index.append([i, j])
    if not edge_index:
        return torch.tensor([[], []], dtype=torch.long)
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

def get_batch_graph(states_np):
    data_list = []
    for state_np_single in states_np:
        # A adjacência é construída sobre o estado de N_AGENT (incluindo mortos/zeros)
        neighbors = adjacency_graphsage(state_np_single.tolist())
        edge_index = neighbors_to_edge_index(neighbors)
        x = torch.tensor(state_np_single, dtype=torch.float32)
        data_list.append(Data(x=x, edge_index=edge_index))
    # PyTorch Geometric cria um super-grafo (Batch)
    return Batch.from_data_list(data_list)

# -------------------
# Modelos
# -------------------
class FeatureMLP(nn.Module):
    def __init__(self, input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class RelationGraphSAGE(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.sage1(x, edge_index))
        x = F.relu(self.sage2(x, edge_index))
        return x

class QNet(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 3, action_dim)
    
    def forward(self, feat, rel1, rel2):
        x = torch.cat([feat, rel1, rel2], dim=-1)
        return self.fc(x)

class DGNAgent(nn.Module):
    def __init__(self, n_agent=N_AGENT, input_dim=REAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.n_agent = n_agent
        self.feature_mlp = FeatureMLP(input_dim, hidden_dim)
        self.relation1 = RelationGraphSAGE(hidden_dim)
        self.relation2 = RelationGraphSAGE(hidden_dim)
        self.q_net = QNet(hidden_dim, action_dim)
        self.input_dim = input_dim
    
    def forward(self, obs, edge_index):
        feat = self.feature_mlp(obs)
        rel1 = self.relation1(feat, edge_index)
        rel2 = self.relation2(rel1, edge_index)
        q = self.q_net(feat, rel1, rel2)
        return q

# -------------------
# Replay Buffer
# -------------------
class ReplayBuffer:
    def __init__(self, capacity=200000):
        self.capacity = capacity
        self.buffer = []
    
    def add(self, s, a, s_next, r, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s, a, s_next, r, done))
    
    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

# -------------------
# Inicialização Ambiente e Agentes
# -------------------
env = parallel_env(map_size=MAP_SIZE, max_cycles=MAX_STEPS, render_mode="none")

agent = DGNAgent()
target_agent = DGNAgent()
target_agent.load_state_dict(agent.state_dict())
optimizer = optim.Adam(agent.parameters(), lr=LR)
buffer = ReplayBuffer()
eps = EPS_START

# -------------------
# Loop de Treino
# -------------------
for ep in range(EPISODES):
    obs = env.reset()
    total_reward = 0
    obs_keys = list(obs.keys())
    agent_handle = obs_keys[0]
    enemy_handle = obs_keys[1]

    for step in range(MAX_STEPS):
        full_obs_agents = obs[agent_handle]
        full_obs_enemies = obs[enemy_handle]

        # Processar estado atual
        state_alive = observation_torch(full_obs_agents, full_obs_enemies)
        num_agents_alive = state_alive.size(0)
        state = torch.zeros(N_AGENT, REAL_INPUT_DIM, dtype=torch.float32)
        if num_agents_alive > 0:
            state[:num_agents_alive] = state_alive

        # Construir grafo
        neighbors = adjacency_graphsage(state.cpu().numpy().tolist())
        edge_index = neighbors_to_edge_index(neighbors)

        # Seleção de Ação (Epsilon-Greedy)
        with torch.no_grad():
            # Q-values para N_AGENT (incluindo agentes mortos/zeros)
            q_values_full = agent(state, edge_index) 

        q_values = q_values_full[:num_agents_alive]
        action_alive = torch.zeros(num_agents_alive, dtype=torch.long)
        for i in range(num_agents_alive):
            # Ação aleatória (exploração) ou melhor ação (explotação)
            action_alive[i] = random.randint(0, ACTION_DIM-1) if random.random() < eps else torch.argmax(q_values[i])

        # Buffer de Ações (N_AGENT)
        action_buffer = torch.zeros(N_AGENT, dtype=torch.long)
        action_buffer[:num_agents_alive] = action_alive

        # Ações do inimigo (aleatórias)
        num_enemies_alive = full_obs_enemies.shape[0] if isinstance(full_obs_enemies, np.ndarray) else len(full_obs_enemies)
        acts_enemy = np.random.randint(0, ACTION_DIM, size=num_enemies_alive, dtype=np.int32)

        live_agent_names = [agent for agent in env.agents if agent.startswith(agent_handle)]
        live_enemy_names = [agent for agent in env.agents if agent.startswith(enemy_handle)]

        # Mapear ações para nomes de agentes
        actions_dict = {}
        for name, action in zip(live_agent_names, action_alive.cpu().numpy().astype(np.int32).tolist()):
            actions_dict[name] = action
        for name, action in zip(live_enemy_names, acts_enemy.tolist()):
            actions_dict[name] = action

        # Passo no ambiente
        next_obs, reward, done, truncated, info = env.step(actions_dict)

        # Processar próximo estado
        next_state_alive = observation_torch(next_obs[agent_handle], next_obs[enemy_handle])
        next_num_agents_alive = next_state_alive.size(0)
        next_state = torch.zeros(N_AGENT, REAL_INPUT_DIM, dtype=torch.float32)
        if next_num_agents_alive > 0:
            next_state[:next_num_agents_alive] = next_state_alive

        # Processar recompensas
        rewards_tensor = torch.zeros(N_AGENT, dtype=torch.float32)
        if live_agent_names:
            rewards_alive_list = [reward.get(name, 0.0) for name in live_agent_names]
            rewards_tensor[:min(len(rewards_alive_list), N_AGENT)] = torch.tensor(rewards_alive_list, dtype=torch.float32)

        # Flags de término do episódio
        agent_terminal = done.get(agent_handle, False) or truncated.get(agent_handle, False)
        enemy_terminal = done.get(enemy_handle, False) or truncated.get(enemy_handle, False)

        # Adicionar ao buffer
        buffer.add(state, action_buffer, next_state, rewards_tensor, agent_terminal)

        # Decaimento do epsilon (AGORA MAIS LENTO)
        eps = max(EPS_END, eps * EPS_DECAY)
        obs = next_obs
        total_reward += sum(reward.values()) if reward else 0

        if agent_terminal or enemy_terminal:
            break

        # -------------------
        # Treinamento (Lógica AJUSTADA para GNN Batch)
        # -------------------
        if len(buffer.buffer) > BATCH_SIZE:
            batch = buffer.sample()
            states = torch.stack([b[0] for b in batch])
            next_states = torch.stack([b[2] for b in batch])

            states_np = states.cpu().numpy()
            next_states_np = next_states.cpu().numpy()

            # Criação dos super-grafos
            batch_graph_s = get_batch_graph(states_np)
            batch_graph_s_next = get_batch_graph(next_states_np)

            states_x, edge_index_s = batch_graph_s.x, batch_graph_s.edge_index
            next_states_x, edge_index_s_next = batch_graph_s_next.x, batch_graph_s_next.edge_index

            # Preparar Ações, Recompensas e Dones no formato de Nó Único (N_Total = BATCH_SIZE * N_AGENT)
            actions = torch.stack([b[1] for b in batch]).view(-1)  # (N_Total)
            
            rewards_flat = torch.stack([b[3] for b in batch]).view(-1) # (N_Total)
            dones_raw = [b[4] for b in batch]
            # Como done é por episódio, replicamos para todos os N_AGENT do episódio
            dones_flat = torch.tensor(dones_raw, dtype=torch.float32).view(BATCH_SIZE, 1).repeat(1, N_AGENT).view(-1) # (N_Total)

            # 1. Q-Value Predito (da rede 'agent')
            # q_pred_all é (N_Total, ACTION_DIM)
            q_pred_all = agent(states_x, edge_index_s)
            # Seleciona o Q-value para a ação escolhida (N_Total)
            q_pred = q_pred_all.gather(1, actions.unsqueeze(1)).squeeze(1) 

            # 2. Q-Value Target (da rede 'target_agent')
            with torch.no_grad():
                # Q-values do próximo estado (N_Total, ACTION_DIM)
                q_next_all = target_agent(next_states_x, edge_index_s_next)
                
                # Max Q next (por nó) (N_Total)
                max_q_next = q_next_all.max(dim=1)[0]
                
                # Target Q (usando a recompensa e done de cada nó)
                # q_target = R + GAMMA * max(Q_next) * (1 - Done)
                q_target = rewards_flat + GAMMA * max_q_next * (1 - dones_flat.float())

            # 3. Cálculo da Perda (Nó por Nó)
            loss = F.mse_loss(q_pred, q_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Soft update da rede target
            for param, target_param in zip(agent.parameters(), target_agent.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            # Log da loss no WandB
            wandb.log({"loss": loss.item(), "episode": ep})

    print(f"Episode {ep}, total_reward={total_reward:.2f}, epsilon={eps:.4f}")
    wandb.log({"total_reward": total_reward, "epsilon": eps, "episode": ep})

    # Salva modelo periodicamente
    if ep % 100 == 0:
        torch.save(agent.state_dict(), "dgn_agent.pth")
        wandb.save("dgn_agent.pth")