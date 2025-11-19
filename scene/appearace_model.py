import torch
from torch import nn
import torch.nn.functional as F

class EmbeddingModel(nn.Module):
    def __init__(
            self,
            appearance_n_fourier_freqs: int = 4,
            sh_degree: int = 3,
            appearance_embedding_dim: int = 16,
            appearance_model_sh: bool = True,
    ):
        super().__init__()

        self.appearance_model_sh = appearance_model_sh
        feat_in = 3
        if self.appearance_model_sh:
            feat_in = ((sh_degree + 1) ** 2) * 3
        else:
            feat_in = 3

        mlp_input_dim = appearance_embedding_dim + feat_in + 6 * appearance_n_fourier_freqs

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 256),           
            nn.ReLU(),
            nn.Linear(256, 128),          
            nn.ReLU(),
            nn.Linear(128, feat_in * 2),   
        )


        self.highlight_branch = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  
            nn.Sigmoid()        
        )


        self.color_bias = nn.Parameter(torch.zeros(3))  


        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, color, image_embed, feature_embed):
        C0 = 0.28209479177387814
        input_color = color
        if not self.appearance_model_sh:
            color = color[..., :3]

        inp = torch.cat((color, image_embed, feature_embed), dim=-1)

        output = self.mlp(inp) * 0.01  

        offset, mul = torch.split(output, [color.shape[-1], color.shape[-1]], dim=-1)

        offset = torch.cat((
            (offset / C0) + self.color_bias[:offset.shape[-1]], 
            torch.zeros_like(input_color[..., offset.shape[-1]:])
        ), dim=-1)

        highlight_mask = self.highlight_branch(inp)  

        mul = mul * (1 + 2 * highlight_mask)  

        mul = mul.repeat(1, input_color.shape[-1] // mul.shape[-1])

        base_result = input_color * mul + offset

        return base_result  
    
class IlluminationNet(nn.Module):       
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, 1, bias=False)  
        self.sigmoid = nn.Sigmoid()
        self.global_bias = nn.Parameter(torch.tensor([1.1, 1.0, 1.0])) 

    def forward(self, depth):
        H, W = depth.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=depth.device),
            torch.linspace(-1, 1, W, device=depth.device),
            indexing='ij'
        )
        r2 = xx ** 2 + yy ** 2
        pos_enc = torch.stack([xx, yy, r2], dim=-1)  

        depth_in = depth.unsqueeze(0).unsqueeze(0)  
        pos_enc_in = pos_enc.permute(2, 0, 1).unsqueeze(0)  
        x = torch.cat([depth_in, pos_enc_in], dim=1) 

        light_map = self.sigmoid(self.conv(x))  
        light_map = light_map * self.global_bias.view(1, 3, 1, 1)
        return light_map.squeeze(0)  


class AttenuationNet(nn.Module):       
    def __init__(self, appearance_dim=16):
        super().__init__()
        self.appearance_dim = appearance_dim
        self.conv = nn.Conv2d(1 + appearance_dim, 6, 1, bias=False)
        nn.init.uniform_(self.conv.weight, 0, 3.0)
        self.beta_coef = nn.Parameter(torch.rand(6, 1, 1) * 0.5)
        self.softplus = nn.Softplus()

    def forward(self, depth, appearance_embed):
        depth = depth.unsqueeze(0).unsqueeze(0) 
        app = appearance_embed.unsqueeze(0)     
        x = torch.cat([depth, app], dim=1)      

        beta_raw = self.softplus(self.conv(x))   
        beta = torch.stack([
            torch.sum(beta_raw[:, 0:2] * torch.sigmoid(self.beta_coef[0:2]), dim=1),
            torch.sum(beta_raw[:, 2:4] * torch.sigmoid(self.beta_coef[2:4]), dim=1),
            torch.sum(beta_raw[:, 4:6] * torch.sigmoid(self.beta_coef[4:6]), dim=1),
        ], dim=1)  

        attenuation = torch.exp(-beta * depth.clamp(min=0))
        return attenuation.squeeze(0)  


class ScatteringNet(nn.Module):      
    def __init__(self):
        super().__init__()
        self.scattering_conv = nn.Conv2d(1, 3, 1, bias=False)
        self.residual_conv = nn.Conv2d(1, 3, 1, bias=False)
        nn.init.uniform_(self.scattering_conv.weight, 0, 1.0)
        nn.init.uniform_(self.residual_conv.weight, 0, 0.5)

        self.B_inf = nn.Parameter(torch.rand(3, 1, 1) * 0.2) 
        self.J_prime = nn.Parameter(torch.rand(3, 1, 1) * 0.1)  
        self.softplus = nn.Softplus()

    def forward(self, depth):
        beta_s = self.softplus(self.scattering_conv(depth.unsqueeze(0)))
        beta_r = self.softplus(self.residual_conv(depth.unsqueeze(0)))

        scatter = (self.B_inf * (1 - torch.exp(-beta_s * depth.unsqueeze(0))) +
                   self.J_prime * torch.exp(-beta_r * depth.unsqueeze(0)))
        return scatter.squeeze(0) 
