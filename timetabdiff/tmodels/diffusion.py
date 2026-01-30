import torch.nn.functional as F
import torch
import math
import numpy as np
from tmodels.noise_schedule import *
from tqdm import tqdm
from itertools import chain

"""
“Our implementation of the continuous-time masked diffusion is inspired by https://arxiv.org/abs/2406.07524's implementation at [https://github.com/kuleshov-group/mdlm], with modifications to support data distributions that include categorical dimensions of different sizes.”
"""

S_churn= 1
S_min=0
S_max=float('inf')
S_noise=1

class TUnifiedCtimeDiffusion(torch.nn.Module):
    def __init__(
            self,
            num_classes: np.array,
            num_numerical_features: int,
            denoise_fn,
            y_only_model,
            num_timesteps=1000,
            scheduler='power_mean',
            cat_scheduler='log_linear',
            noise_dist='uniform',
            edm_params={},
            noise_dist_params={},
            noise_schedule_params={},
            sampler_params={},
            device=torch.device('cpu'),
            **kwargs
        ):

        super(TUnifiedCtimeDiffusion, self).__init__()

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes # it as a vector [K1, K2, ..., Km]
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([np.repeat(i, num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes)>0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0 
        self.num_classes_w_mask = tuple([k + 1 for k in self.num_classes])

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = []
        for i in range(1, len(offsets)):
            self.slices_for_classes.append(np.arange(offsets[i - 1], offsets[i]))
        self.offsets = torch.from_numpy(offsets).to(device)
        
        offsets = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes)+1)
        offsets = np.append([0], offsets)
        self.slices_for_classes_with_mask = []
        for i in range(1, len(offsets)):
            self.slices_for_classes_with_mask.append(np.arange(offsets[i - 1], offsets[i]))

        self._denoise_fn = denoise_fn
        self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params
        if self.num_numerical_features == 0:
            self.sampler_params['stochastic_sampler'] = False
            self.sampler_params['second_order_correction'] = False
        
        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []
        
        self.device = device
        
        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical = num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")
        
        if self.cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif self.cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories = len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f"The noise schedule--{self.cat_scheduler}-- is not implemented for discrete data at CTIME ")

    def mixed_loss(self, x):
        """
        x: [B, S, Total_Features] - 3D 구조 유지
        """
        b, s, _ = x.shape
        x_num = x[:, :, :self.num_numerical_features]
        x_cat = x[:, :, self.num_numerical_features:].long()

        if self.noise_dist == "uniform_t":
            t = torch.rand(b, device=self.device).view(b, 1)
            sigma_num = self.num_schedule.total_noise(t)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)
        else:
            sigma_num = self.sample_ctime_noise(x)
            t = self.num_schedule.inverse_to_t(sigma_num)
            while torch.any((t < 0) + (t > 1)):     
                invalid_idx = ((t < 0) + (t > 1)).nonzero().squeeze(-1)
                sigma_num[invalid_idx] = self.sample_ctime_noise(x[:len(invalid_idx)])
                t = self.num_schedule.inverse_to_t(sigma_num)
            t = t.view(b, 1); sigma_num = sigma_num.view(b, 1)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)

        sn_3d = sigma_num.unsqueeze(-1)
        sc_3d = sigma_cat.unsqueeze(-1)
        x_num_t = x_num + torch.randn_like(x_num) * sn_3d if x_num.shape[-1] > 0 else x_num
        x_num_t = x_num_t.clone().detach().requires_grad_(True)
        
        x_cat_t, x_cat_t_soft = (x_cat, x_cat)
        if x_cat.shape[-1] > 0:
            move_3d = -torch.expm1(-sc_3d)
            x_cat_t, x_cat_t_soft = self.q_xt_3d(x_cat, move_3d, strategy='soft' if 'per_column' in self.cat_scheduler else 'hard')

        model_out_num, model_out_cat = self._denoise_fn(x_num_t, x_cat_t_soft, t.squeeze())
        # # ===== 디버깅 =====
        # with torch.no_grad():
        #     print("\n[DEBUG mixed_loss]")
        #     print(f"t range: [{t.min().item():.4e}, {t.max().item():.4e}]")
        #     print(f"sigma_num range: [{sigma_num.min().item():.4e}, {sigma_num.max().item():.4e}]")
        #     print(f"sigma_cat range: [{sigma_cat.min().item():.4e}, {sigma_cat.max().item():.4e}]")

        #     print(f"x_num range: [{x_num.min().item():.4e}, {x_num.max().item():.4e}]")
        #     print(f"x_num_t range: [{x_num_t.min().item():.4e}, {x_num_t.max().item():.4e}]")

        #     print(f"model_out_num range: [{model_out_num.min().item():.4e}, {model_out_num.max().item():.4e}]")

        #     diff = (model_out_num - x_num)
        #     print(f"(pred - target) range: [{diff.min().item():.4e}, {diff.max().item():.4e}]")

        #     if sn_3d.numel() > 0:
        #         scaled = diff / (sn_3d + 1e-8)
        #         print(f"(pred-target)/sigma range: [{scaled.min().item():.4e}, {scaled.max().item():.4e}]")
        
        c_loss = self._edm_loss_3d(model_out_num, x_num, sn_3d) if x_num.shape[-1] > 0 else torch.zeros(1, device=self.device, requires_grad=True)
        d_loss = torch.zeros(1, device=self.device, requires_grad=True)
        if x_cat.shape[-1] > 0:
            logits = self._subs_parameterization_3d(model_out_cat, x_cat_t)
            d_loss = self._absorbed_closs_3d(logits, x_cat, sc_3d, dsigma_cat.unsqueeze(-1))
            
        return d_loss.mean(), c_loss.mean()

    def _edm_loss_3d(self, D_yn, y, sigma_3d):
        """
        D_yn, y: [B, S, F_num]
        sigma_3d: [B, 1, 1] (또는 [B, 1, F_num])
        """
        # EDM Weighting 공식 적용
        weight = (sigma_3d ** 2 + self.edm_params['sigma_data'] ** 2) / \
                 (sigma_3d * self.edm_params['sigma_data']) ** 2
        
        # [B, S, F] 전체에 대해 MSE 계산
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def _absorbed_closs_3d(self, model_output, x0, sigma_3d, dsigma_3d):
        """
        model_output (logits): [B, S, F_cat, Max_K]
        x0: [B, S, F_cat]
        """
        # 정답 카테고리의 log-probability 추출
        log_p_theta = torch.gather(model_output, -1, x0.unsqueeze(-1)).squeeze(-1)
        
        # MDLM 공식에 따른 ELBO Weighting
        elbo_weight = - dsigma_3d / torch.expm1(sigma_3d)
        loss = elbo_weight * log_p_theta
        return loss

    # --- 2. 샘플링 엔진 (3D) ---
    @torch.no_grad()
    def sample(self, num_samples, seq_len):
        b, s = num_samples, seq_len
        t = torch.linspace(0, 1, self.num_timesteps, device=self.device).view(-1, 1)
        
        sn = self.num_schedule.total_noise(t)
        sc = self.cat_schedule.total_noise(t)
        sn_next = torch.cat([torch.zeros(1, 1, device=self.device), sn[:-1]])
        sc_next = torch.cat([torch.zeros(1, 1, device=self.device), sc[:-1]])
        
        z_norm = torch.randn((b, s, self.num_numerical_features), device=self.device) * sn[-1]
        z_cat = self._sample_masked_prior(b, s)

        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            z_norm, z_cat, _ = self.edm_update_3d(z_norm, z_cat, i, t[i], t[i-1] if i > 0 else None, t[i], sn[i], sn_next[i], sn[i], sc[i], sc_next[i], sc[i], s)
        return torch.cat([z_norm, z_cat], dim=-1).cpu()

    def edm_update_3d(self, x_num, x_cat, i, t_cur, t_next, t_hat, sn_cur, sn_next, sn_hat, sc_cur, sc_next, sc_hat, s):
        b = x_num.shape[0]
        cfg = self.y_only_model is not None
        
        x_num_hat = x_num + (sn_hat**2 - sn_cur**2).sqrt() * S_noise * torch.randn_like(x_num)
        move_chance = -torch.expm1(sc_cur - sc_hat).view(1, 1, 1)
        x_cat_hat, _ = self.q_xt_3d(x_cat, move_chance) if len(self.num_classes) > 0 else (x_cat, None)
        
        x_cat_hat_oh = self.to_one_hot(x_cat_hat.view(b*s, -1)).float().view(b, s, -1) if len(self.num_classes) > 0 else x_cat_hat
        denoised, raw_logits = self._denoise_fn(x_num_hat, x_cat_hat_oh, t_hat.repeat(b))

        # [CFG 3D 반영]
        if cfg:
            is_bin = len(self.num_mask_idx) == 0
            is_learnable = self.scheduler == "power_mean_per_column"
            sigma_cond = sn_hat if not is_learnable else ((0.002**(1/7) + t_hat*(80**(1/7)-0.002**(1/7))).pow(7) if is_bin else sn_hat[self.num_mask_idx])
            
            y_num_hat = x_num_hat[:, :, self.num_mask_idx]
            idx_cat = list(chain(*[self.slices_for_classes_with_mask[idx] for idx in self.cat_mask_idx]))
            y_cat_hat = x_cat_hat_oh[:, :, idx_cat]
            
            y_denoised, y_logits = self.y_only_model(y_num_hat, y_cat_hat, t_hat.repeat(b), sigma=sigma_cond.view(1, -1).repeat(b, 1))
            
            denoised[:, :, self.num_mask_idx] = (1 + self.w_num) * denoised[:, :, self.num_mask_idx] - self.w_num * y_denoised
            m_idx = np.concatenate([self.slices_for_classes_with_mask[idx] for idx in self.cat_mask_idx]) if len(self.cat_mask_idx)>0 else []
            if len(m_idx) > 0: raw_logits[:, :, m_idx] = (1 + self.w_cat) * raw_logits[:, :, m_idx] - self.w_cat * y_logits

        d_cur = (x_num_hat - denoised) / sn_hat
        x_num_next = x_num_hat + (sn_next - sn_hat) * d_cur
        
        x_cat_next = x_cat
        if len(self.num_classes) > 0:
            logits = self._subs_parameterization_3d(raw_logits, x_cat_hat)
            at, as_ = torch.exp(-sc_hat).repeat(b*s, 1), torch.exp(-sc_next).repeat(b*s, 1)
            x_cat_next, _ = self._mdlm_update(logits.view(b*s, len(self.num_classes), -1), x_cat_hat.view(b*s, -1), at, as_)
            x_cat_next = x_cat_next.view(b, s, -1)
        return x_num_next, x_cat_next, None
        

    def sample_all(self, num_samples, seq_len, batch_size, keep_nan_samples=False):
        all_s = []
        num_g = 0
        while num_g < num_samples:
            sample = self.sample(batch_size, seq_len)
            mask = torch.any(sample.isnan(), dim=(1,2))
            sample = sample[~mask] if not keep_nan_samples else sample * (~mask).view(-1,1,1).float()
            all_s.append(sample)
            num_g += sample.shape[0]
        return torch.cat(all_s, dim=0)[:num_samples]
    
    def q_xt(self, x, move_chance, strategy='hard'):
        """Computes the noisy sample xt.

        Args:
        x: int torch.Tensor with shape (batch_size,
            diffusion_model_input_length), input. 
        move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        if strategy == 'hard':
            move_indices = torch.rand(
            * x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index, x)
            xt_soft = self.to_one_hot(xt).to(move_chance.dtype)
            return xt, xt_soft
        # elif strategy == 'soft':
        #     bs = x.shape[0]
        #     xt_soft = torch.zeros(bs, torch.sum(self.mask_index+1), device=x.device)
        #     xt = torch.zeros_like(x)
        #     for i in range(len(self.num_classes)):
        #         slice_i = self.slices_for_classes_with_mask[i]
        #         # set the bernoulli probabilities, which determines the "coin flip" transition to the mask class
        #         prob_i = torch.zeros(bs, 2, device=x.device)
        #         prob_i[:,0] = 1-move_chance[:,i]
        #         prob_i[:,-1] = move_chance[:,i]
        #         log_prob_i = torch.log(prob_i)
        #         # draw soft samples and place them back to the corresponding columns
        #         soft_sample_i = F.gumbel_softmax(log_prob_i, tau=0.01, hard=True)
        #         idx = torch.stack((x[:,i]+slice_i[0], torch.ones_like(x[:,i])*slice_i[-1]), dim=-1)
        #         xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
        #         # retrieve the hard samples
        #         xt[:, i] = torch.where(soft_sample_i[:,1] > soft_sample_i[:,0], self.mask_index[i], x[:,i])
        #     return xt, xt_soft
        elif strategy == 'soft':
            bs, f = x.shape
            xt_soft = torch.zeros(bs, sum(self.num_classes_w_mask), device=x.device, dtype=torch.float)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                prob_i = torch.stack([1 - move_chance[:, i], move_chance[:, i]], dim=-1)
                soft_sample_i = F.gumbel_softmax(prob_i.log(), tau=0.01, hard=False)
                xt_soft[:, slice_i[0]:slice_i[-1]+1] = soft_sample_i
            return xt, xt_soft.requires_grad_(True)
    
    def q_xt_3d(self, x, move_chance, strategy='hard'):
        b, s, f = x.shape
        x_flat = x.reshape(b * s, f)
        m_flat = move_chance.expand(b, s, f).reshape(b * s, f)
        xt, xt_s = self.q_xt(x_flat, m_flat, strategy)
        return xt.view(b, s, f), xt_s.view(b, s, -1)
    
    def _subs_parameterization(self, unormalized_prob, xt):
        # log prob at the mask index = - infinity
        unormalized_prob = self.pad(unormalized_prob, self.neg_infinity)
        
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        
        # Take log softmax on the unnormalized probabilities to the logits
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1,
                                        keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)    # (bs, K)
        logits[unmasked_indices] = self.neg_infinity 
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits
    
    def _subs_parameterization_3d(self, prob, xt):
        b, s, f = xt.shape
        logits = self._subs_parameterization(prob.reshape(b*s, -1), xt.reshape(b*s, -1))
        return logits.view(b, s, f, -1)

    def pad(self, x, pad_value):
        """
        Converts a concatenated tensor of class probabilities into a padded matrix, 
        where each sub-tensor is padded along the last dimension to match the largest 
        category size (max number of classes).

        Args:
            x (Tensor): The input tensor containing concatenated probabilities for all the categories in x_cat. 
                        [bs, sum(num_classes_w_mask)]
            pad_value (float): The value filled into the dummy entries, which are padded to ensure all sub-tensors have equal size 
                            along the last dimension.

        Returns:
            Tensor: A new tensorwith
                    [bs, len(num_classes_w_mask), max(num_classes_w_mask)), num_categories]
        """
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [
            torch.cat((
                t, 
                pad_value*torch.ones(*(t.shape[:-1]), max_K-t.shape[-1], dtype=t.dtype, device=t.device)
            ), dim=-1) 
        for t in splited]
        out = torch.stack(padded_, dim=-2)
        return out
    
    def to_one_hot(self, x_cat):
        x_cat_oh = torch.cat(
            [F.one_hot(x_cat[:, i], num_classes=self.num_classes[i]+1,) for i in range(len(self.num_classes))], 
            dim=-1
        )
        return x_cat_oh
    
    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        """
            alpha: (bs,)
        """
        log_p_theta = torch.gather(
            model_output, -1, x0[:, :, None]
        ).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ['log_linear_unified', 'log_linear_per_column']:
            elbo_weight = - dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1/(1-alpha)
        
        loss = elbo_weight * log_p_theta
        return loss
    
    def _sample_masked_prior(self, batch_size, seq_len):
        # return self.mask_index[None,:] * torch.ones(    
        # * batch_dims, dtype=torch.int64, device=self.mask_index.device)
        return self.mask_index[None, None, :].expand(batch_size, seq_len, -1).clone()   
        
    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        """
            # t: (bs,)
            log_p_x0: (bs, K, K_max)
            # alpha_t: (bs,)
            # alpha_s: (bs,)
            alpha_t: (bs, 1/K_cat)
            alpha_s: (bs,1/K_cat)
        """
        move_chance_t = 1 - alpha_t
        move_chance_s = 1 - alpha_s     
        move_chance_t = move_chance_t.unsqueeze(-1)
        move_chance_s = move_chance_s.unsqueeze(-1)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        # There is a noremalizing term is (1-\alpha_t) who's responsility is to ensure q_xs is normalized. 
        # However, omiting it won't make a difference for the Gumbel-max sampling trick in  _sample_categorical()
        q_xs = log_p_x0.exp() * (move_chance_t
                                - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        
        # Important: make sure that prob of dummy classes are exactly 0
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index+1))] for mask_idx in self.mask_index], device=q_xs.device)
        dummy_mask = torch.ones_like(q_xs) * dummy_mask
        q_xs *= dummy_mask
        
        _x = self._sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        
        z_cat = copy_flag * x + (1 - copy_flag) * _x
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = (
            1e-10
            - (torch.rand_like(categorical_probs) + 1e-10).log())
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    
    def sample_ctime_noise(self, batch):
        if self.noise_dist == 'log_norm':
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            sigma = (rnd_normal * self.noise_dist_params['P_std'] + self.noise_dist_params['P_mean']).exp()
        else:
            raise NotImplementedError(f"The noise distribution--{self.noise_dist}-- is not implemented for CTIME ")
        return sigma

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
    
        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss
    
    def edm_update(
            self, x_num_cur, x_cat_cur, i, 
            t_cur, t_next, t_hat,
            sigma_num_cur, sigma_num_next, sigma_num_hat, 
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat, 
        ):
        """
        i = T-1,...,0
        """
        cfg = self.y_only_model is not None
        
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0
        
        # Get x_num_hat by move towards the noise by a small step
        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * S_noise * torch.randn_like(x_num_cur)
        # Get x_cat_hat
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)    # the incremental move change is 1 - alpha_t/alpha_s = 1 - exp(sigma_s - sigma_t)
        x_cat_hat, _ = self.q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)

        # Get predictions
        x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised, raw_logits = self._denoise_fn(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
        )
        
        # Apply cfg updates, if is in cfg mode
        is_bin_class = len(self.num_mask_idx) == 0
        is_learnable = self.scheduler=="power_mean_per_column"
        if cfg:
            if not is_learnable:
                sigma_cond = sigma_num_hat
            else:
                if is_bin_class:
                    sigma_cond = (0.002 ** (1/7) + t_hat * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                else:
                    sigma_cond = sigma_num_hat[self.num_mask_idx]
            y_num_hat = x_num_hat.float()[:, self.num_mask_idx]
            idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
            y_cat_hat = x_cat_hat_oh[:,idx]
            y_only_denoised, y_only_raw_logits = self.y_only_model(
                y_num_hat, 
                y_cat_hat,
                t_hat.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
            )
            
            denoised[:, self.num_mask_idx] *= 1 + self.w_num
            denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
            
            mask_logit_idx = [self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]
            mask_logit_idx = np.concatenate(mask_logit_idx) if len(mask_logit_idx)>0 else np.array([])
            
            raw_logits[:, mask_logit_idx] *= 1 + self.w_cat
            raw_logits[:, mask_logit_idx] -= self.w_cat*y_only_raw_logits
        
        # Euler step
        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur
        
        # Unmasking
        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b,1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b,1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)
        
        # Apply 2nd order correction.
        if self.sampler_params['second_order_correction']:
            if i > 0:
                x_cat_hat_oh = self.to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
                denoised, raw_logits = self._denoise_fn(
                    x_num_next.float(), x_cat_hat_oh,
                    t_next.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b,1)
                )
                if cfg:
                    if not is_learnable:
                        sigma_cond = sigma_num_next
                    else:
                        if is_bin_class:
                            sigma_cond = (0.002 ** (1/7) + t_next * (80 ** (1/7) - 0.002 ** (1/7))).pow(7)
                        else:
                            sigma_cond = sigma_num_next[self.num_mask_idx]
                    y_num_next = x_num_next.float()[:, self.num_mask_idx]
                    idx = list(chain(*[self.slices_for_classes_with_mask[i] for i in self.cat_mask_idx]))
                    y_cat_hat = x_cat_hat_oh[:, idx]
                    y_only_denoised, y_only_raw_logits = self.y_only_model(
                        y_num_next,
                        y_cat_hat,
                        t_next.squeeze().repeat(b), sigma=sigma_cond.unsqueeze(0).repeat(b,1)  # sigma accepts (bs, K_num)
                    )
                    denoised[:, self.num_mask_idx] *= 1 + self.w_num
                    denoised[:, self.num_mask_idx] -= self.w_num*y_only_denoised
                
                d_prime = (x_num_next - denoised) / sigma_num_next
                x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_num_next, x_cat_next, q_xs


    @torch.no_grad()
    def sample_impute(self, x_num, x_cat, num_mask_idx, cat_mask_idx, resample_rounds, impute_condition, w_num, w_cat):
        """
        Modified for 3D Time-series Tabular Data
        x_num: [B, S, F_num], x_cat: [B, S, F_cat]
        """
        self.w_num = w_num
        self.w_cat = w_cat
        self.num_mask_idx = num_mask_idx
        self.cat_mask_idx = cat_mask_idx
        
        b, s, _ = x_num.shape # 시퀀스 길이(s) 추출
        device = self.device
        dtype = torch.float32

        # 1. 마스크 생성 및 차원 확장 [1, 1, F] 형태로 만들어 브로드캐스팅 지원
        num_mask = torch.tensor([i in num_mask_idx for i in range(self.num_numerical_features)], 
                                device=device, dtype=dtype).view(1, 1, -1)
        cat_mask = torch.tensor([i in cat_mask_idx for i in range(len(self.num_classes))], 
                                device=device, dtype=dtype).view(1, 1, -1)

        # 2. Time 및 Noise Chain 생성 (기존 로직 유지)
        t = torch.linspace(0, 1, self.num_timesteps, dtype=dtype, device=device).view(-1, 1)
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        
        sigma_num_next = torch.cat([torch.zeros(1, 1, device=device), sigma_num_cur[:-1]])
        sigma_cat_next = torch.cat([torch.zeros(1, 1, device=device), sigma_cat_cur[:-1]])
        
        # Stochastic sampler (t_hat, sigma_hat) 계산
        if self.sampler_params.get('stochastic_sampler', False):
            gamma = min(S_churn / self.num_timesteps, math.sqrt(2) - 1) * (S_min <= sigma_num_cur) * (sigma_num_cur <= S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat, sigma_num_hat, sigma_cat_hat = t, sigma_num_cur, sigma_cat_cur

        # 3. 초기 값 설정 (3D)
        if impute_condition == "x_t":
            # 알려진 값에 최대 노이즈를 섞어서 시작
            z_norm = x_num + torch.randn_like(x_num) * sigma_num_cur[-1]
            z_cat = self._sample_masked_prior(b, s) # [B, S, F_cat]를 모두 [MASK]로
        else: # "x_0"
            z_norm = x_num
            z_cat = x_cat
            
        # 4. Imputation Loop
        pbar = tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc="Imputing")
        for i in pbar:
            for u in range(resample_rounds):
                # [Known Part] 현재 시점의 노이즈 강도로 원본(알려진 값)을 오염시킴
                if impute_condition == "x_t":
                    z_norm_known = x_num + torch.randn_like(x_num) * sigma_num_next[i]
                    # 범주형 move_chance 계산
                    m_chance = 1 - torch.exp(-sigma_cat_next[i]) if i < (self.num_timesteps-1) else torch.ones_like(sigma_cat_next[i])
                    z_cat_known, _ = self.q_xt_3d(x_cat, m_chance.view(1, 1, 1))
                else:
                    z_norm_known, z_cat_known = x_num, x_cat
                
                # [Unknown Part] 역과정(Denoising) 한 스텝 수행
                z_norm_un, z_cat_un, _ = self.edm_update_3d(
                    z_norm, z_cat, i, 
                    t[i], t[i-1] if i > 0 else None, t_hat[i],
                    sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i], 
                    sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i], s
                )
                
                # [Merge] 마스크를 사용하여 알려진 값과 예측된 값을 결합
                z_norm = (1 - num_mask) * z_norm_known + num_mask * z_norm_un
                z_cat = (1 - cat_mask.long()) * z_cat_known + cat_mask.long() * z_cat_un

                # [Resampling] Langevin 다이나믹스를 위한 추가 노이즈 주입 (u > 0 일 때)
                if u < resample_rounds - 1:
                    z_norm += (sigma_num_cur[i]**2 - sigma_num_next[i]**2).sqrt() * S_noise * torch.randn_like(z_norm)
                    m_chance_resample = -torch.expm1(sigma_cat_next[i] - sigma_cat_cur[i])
                    z_cat, _ = self.q_xt_3d(z_cat, m_chance_resample.view(1, 1, 1))
        
        return torch.cat([z_norm, z_cat], dim=-1).cpu()