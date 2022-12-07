import torch
from torch import nn

def flatten_seq(x):
    return torch.reshape(x, (-1,) + x.shape[2:])

def unflatten_seq(x, first_dim):
    return torch.reshape(x, (first_dim, -1,) + x.shape[1:])

def initialize_weights_tf2(m):
    """Same weight initializations as tf2"""
    if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if type(m) == nn.GRUCell:
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)
        
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class DreamerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        image_size = 64  # assume 64 like dreamer
        depth = 32
        act = nn.ReLU(inplace=True)

        def out_fn(x):
            return 1 + int((x - 4) / 2)

        out_size = image_size
        for _ in range(4):
            out_size = out_fn(out_size)
        self.out_dim = depth * 8 * out_size * out_size  # 1536

        # Define Network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, depth, kernel_size=4, stride=2),
            act,
            nn.Conv2d(depth, depth * 2, kernel_size=4, stride=2),
            act,
            nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2),
            act,
            nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2),
            act,
            nn.Flatten(),
        ).apply(initialize_weights_tf2)

    def forward(self, x):
        # x = (T, B, C, H, W)
        T = x.size(0)
        x = x / 255.0 - 0.5
        flattened = flatten_seq(x)
        embed = self.encoder(flattened)
        return unflatten_seq(embed, T)
    
class DrQv2Encoder(nn.Module):
    def __init__(self, img_size=64, in_channel=3):
        super().__init__()

        assert img_size == 64, 'not supported other stuff'
        # self.repr_dim = 32 * 35 * 35
        self.repr_dim = 20000

        self.convnet = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.apply(weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        return h

class BYOLWorldModel(nn.Module):
    def __init__(self,
                 obs_shape,
                 action_dim,
                 use_dreamer,
                 latent_size,
                 predictor_hidden_dim,
                 predictor_dim):
        super().__init__()
        self.latent_size = latent_size
        self.action_dim = action_dim
        
        use_img = len(obs_shape) == 3
        if use_img:
            if use_dreamer:
                self.encoder = DreamerEncoder()
            else:
                self.encoder = DrQv2Encoder()
        else:
            # TODO don't hardcode
            self.encoder = nn.Sequential(
                nn.Linear(obs_shape[0], 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256)
            )
        
        repr_dim = (4096 if use_dreamer else 20000) if use_img else 256
        self.closed_gru = nn.GRUCell(repr_dim + action_dim, latent_size)
        self.open_gru = nn.GRUCell(repr_dim, latent_size)
        
        self.predictor = nn.Sequential(
            nn.Linear(latent_size, predictor_hidden_dim),
            nn.LayerNorm(predictor_hidden_dim),
            nn.Tanh(),
            nn.Linear(predictor_hidden_dim, predictor_hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(predictor_hidden_dim, predictor_dim)
        )
    
    def forward(self, seq_states, seq_actions):
        T, B = seq_states.size(0), seq_states.size(1)
        
        seq_states_flat = flatten_seq(seq_states)
        embeddings = self.encoder(seq_states_flat)
        embeddings = unflatten_seq(embeddings, T)
        state = torch.zeros(B, self.latent_size + self.action_dim)
        
        states = []
        first_embed = embeddings[0]
        first_action = seq_actions[0]
        first_input = torch.cat([first_embed, first_action], dim=-1)
        state = self.closed_gru(first_input, state)
        states.append(state)
        
        for action in seq_actions[1:]:
            state = self.open_gru(action, state)
            states.append(state)
        
        states = torch.stack(states, dim=0)
        latents = self.predictor(states)
        
        return latents, embeddings
    
    def compute_uncertainty(self, seq_states, seq_actions):
        '''BYOL-Explore uncertainty measurement.'''