from typing import Callable, Union

from models.transformer import Reconstructor, Tokenizer, Transformer
import torch
import torch.nn as nn
import torch.optim
import math

ModuleType = Union[str, Callable[..., nn.Module]]

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_token, max_seq_len=1000):
        super().__init__()
        # d_token은 짝수 (sin, cos 쌍을 맞춰야 함)
        assert d_token % 2 == 0, "d_token must be even for RoPE"
        self.d_token = d_token
               
        theta = 1.0 / (10000.0 ** (torch.arange(0, d_token, 2).float() / d_token))
        
        # 시간축 인덱스 (0, 1, 2, ..., max_seq_len)
        seq_idx = torch.arange(max_seq_len).float()
        
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        
        # [seq_len, d_token] 형태로 sin, cos 생성
        # 각 각도에 대해 sin, cos를 두 번씩 반복해서 d_token 길이를 맞춤
        emb = torch.cat([idx_theta, idx_theta], dim=-1)
                
        self.register_buffer('cos', emb.cos()) # [max_seq_len, d_token]
        self.register_buffer('sin', emb.sin()) # [max_seq_len, d_token]

    def _rotate_half(self, x):
        # x의 절반을 쪼개서 회전 행렬 연산을 위한 준비
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Num_Features, d_token]
        seq_len = x.size(1)
        
        # 현재 입력된 시퀀스 길이에 맞춰 cos, sin 슬라이싱 및 차원 확장
        # [1, Seq_Len, 1, d_token] 형태로 만들어 브로드캐스팅 가능하게 함
        cos = self.cos[:seq_len, :].view(1, seq_len, 1, self.d_token)
        sin = self.sin[:seq_len, :].view(1, seq_len, 1, self.d_token)
        
        # RoPE 공식 적용: x * cos(m*theta) + rotate_half(x) * sin(m*theta)
        return (x * cos) + (self._rotate_half(x) * sin)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t = 512, use_mlp=True, dim_parents=[]):
        super().__init__()
        self.dim_t = dim_t

        self.proj = nn.Linear(d_in, dim_t)

        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)

        self.map_noise = PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        
        self.use_mlp = use_mlp

        if math.prod(dim_parents)> 0:
             self.parent_projs = nn.ModuleList([
                nn.Linear(dim_p, dim_t) for dim_p in dim_parents
            ])
        else:
            self.parent_projs = None
    
    def forward(self, x, timesteps, parents=None):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        emb = self.time_embed(emb)        
        if self.parent_projs is not None and parents is not None:
            for i, parent in enumerate(parents):
                emb = emb + self.parent_projs[i](parent)

        x = self.proj(x) + emb
        return self.mlp(x)

    
class TUniModMLP(nn.Module):
    def __init__(
            self, d_numerical, categories, num_layers, d_token,
            n_head = 1, factor = 4, bias = True, dim_t=512, use_mlp=True, 
            max_seq_len=100, **kwargs
        ):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        
        self.rope = RotaryPositionalEmbedding(d_token, max_seq_len)

        self.tokenizer = Tokenizer(d_numerical, categories, d_token, bias = bias)
        self.encoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        
        # 2. MLP 입력 차원: (피처 개수 * d_token)
        # 시계열이라도 각 시점(token)의 차원을 유지하며 처리하기 위함
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
        
        self.decoder = Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = Reconstructor(d_numerical, categories, d_token)

    def forward(self, x_num, x_cat, timesteps):
        B, S, _ = x_num.shape
        num_features = self.d_numerical + len(self.categories)

        # [단계 1] 토큰화
        x_num_flat = x_num.reshape(B * S, -1)
        x_cat_flat = x_cat.reshape(B * S, -1) if x_cat is not None else None
        e = self.tokenizer(x_num_flat, x_cat_flat) # [B*S, Features+1, d_token]
        e = e[:, 1:, :] # CLS 제거

        # [단계 2] 시계열 복원 및 RoPE
        e = e.view(B, S, num_features, self.d_token)
        e = self.rope(e) 

        # [단계 3] 핵심: 시계열 전체를 하나의 시퀀스로 묶어 Encoder 통과
        e_all = e.reshape(B, S * num_features, self.d_token)
        y_all = self.encoder(e_all)
        
        # [단계 4] MLP (Diffusion timestep 주입)        
        y_all = y_all.view(B, S, num_features, self.d_token)
        
        # MLP 입력을 위해 [B*S, F*d_token]으로 변경
        y_flat = y_all.reshape(B * S, num_features * self.d_token)        
        pred_y_flat = self.mlp(y_flat, timesteps.repeat_interleave(S))
        
        # [단계 5] Decoder (다시 시계열 문맥을 고려하며 디코딩)
        pred_y_all = pred_y_flat.view(B, S * num_features, self.d_token)
        pred_e_all = self.decoder(pred_y_all)
        
        # [단계 6] 복원 및 리쉐이프
        pred_e_flat = pred_e_all.reshape(B * S, num_features, self.d_token)
        x_num_pred, x_cat_pred = self.detokenizer(pred_e_flat)

        x_num_pred = x_num_pred.view(B, S, -1)
        if len(self.categories) > 0:
            x_cat_pred = torch.cat(x_cat_pred, dim=-1).view(B, S, -1)
        else:
            x_cat_pred = torch.zeros_like(x_cat).to(x_num_pred.dtype)

        return x_num_pred, x_cat_pred


class Precond(nn.Module):
    def __init__(self,
        denoise_fn,
        sigma_data = 0.5,
        net_conditioning = "sigma",
    ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):
        # x_num: [B, S, F], x_cat: [B, S, F_cat] (3D 시계열 입력)
        device = x_num.device
        x_num = x_num.to(torch.float32)
        sigma = sigma.to(torch.float32) # [B, 1] 또는 [B, F]

        # 1. sigma_cond 계산 (원본 로직 유지)
        # sigma가 2차원[B, F]인 경우(컬럼별 스케줄링) 고정 스케줄 적용
        if sigma.ndim > 1 and sigma.shape[-1] > 1:
            sigma_cond = (0.002 ** (1/7) + t * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
        else:
            sigma_cond = sigma 
        
        # 2. EDM 스케일링 계수 계산
        # sigma: [B, 1] -> [B, 1, 1]로 변환하여 x_num [B, S, F]와 연산 가능하게 함
        # 피처별 시그마[B, F]일 경우 [B, 1, F]로 변환
        s_target = sigma.view(sigma.shape[0], 1, -1) 
        
        c_skip = self.sigma_data ** 2 / (s_target ** 2 + self.sigma_data ** 2)
        c_out = s_target * self.sigma_data / (s_target ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + s_target ** 2).sqrt()
        c_noise = sigma_cond.log() / 4 # [B] 또는 [B, 1]

        # 3. 입력값 스케일링 및 모델 호출
        x_in = c_in * x_num # [B, S, F]
        
        if self.net_conditioning == "sigma":
            # c_noise는 [B] 형태로 평탄화하여 전달
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, c_noise.flatten())
        elif self.net_conditioning == "t":
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, t)

        # 4. 결과 복원 (Denoising 복구 공식)
        # F_x: [B, S, F]
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        
        return D_x, x_cat_pred

class Model(nn.Module):
    def __init__(
            self, denoise_fn,
            sigma_data=0.5, 
            precond=False, 
            net_conditioning="sigma",
            **kwargs
        ):
        super().__init__()
        self.precond = precond
        if precond:
            self.denoise_fn_D = Precond(
                denoise_fn,
                sigma_data=sigma_data,
                net_conditioning=net_conditioning
            )
        else:
            self.denoise_fn_D = denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        # x_num, x_cat: [B, S, F]
        if self.precond:
            # Precond 모드일 때는 sigma가 반드시 필요함
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        else:
            return self.denoise_fn_D(x_num, x_cat, t)