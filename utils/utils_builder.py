from cgi import test
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from resnet1d import ResNet18, ResNet34, ResNet50, ResNet101
from vit1d import vit_base, vit_small, vit_tiny, vit_middle

def center_kernel(K):
    # double center: Kc = K - row_mean - col_mean + total_mean
    n = K.size(0)
    row_mean = K.mean(dim=1, keepdim=True)   # [n,1]
    col_mean = K.mean(dim=0, keepdim=True)   # [1,n]
    total_mean = K.mean()
    Kc = K - row_mean - col_mean + total_mean
    return Kc  

class SteinKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_type = 'rbf'
        self.adaptive_bandwidth = True
        self.min_bandwidth = 0.1
        self.max_bandwidth = 10.0
        self.bandwidth_factor = 1.0
        self.register_buffer('bandwidth', torch.tensor(1.0))
        self.chunk_size = 256

    def update_bandwidth(self, x, y):
        with torch.no_grad():
            sample_size = x.size(0)#min(512, x.size(0))
            x_sample = x[:sample_size]
            y_sample = y[:sample_size]
            
            diff = x_sample.unsqueeze(1) - y_sample.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            
            bandwidth_sq = torch.median(dist_sq.view(-1))
            bandwidth = torch.sqrt(bandwidth_sq / 2.0) * self.bandwidth_factor
            
            self.bandwidth = torch.clamp(
                bandwidth,
                min=self.min_bandwidth,
                max=self.max_bandwidth
            )
            
    def rbf_kernel(self, x, y):
        if self.training:
            self.update_bandwidth(x,y)

        batch_size = x.size(0)
        kernel_matrix = torch.zeros(batch_size, y.size(0), device=x.device)

        for i in range(0, batch_size, self.chunk_size):
            end_i = min(i + self.chunk_size, batch_size)
            chunk_x = x[i:end_i]

            for j in range(0, y.size(0), self.chunk_size):
                end_j = min(j+self.chunk_size, batch_size)
                chunk_y = y[j:end_j]

                diff = chunk_x.unsqueeze(1) - chunk_y.unsqueeze(0)
                dist_sq = torch.sum(diff ** 2, dim=-1)

                if self.bandwidth is None:
                    bandwidth_sq = torch.median(dist_sq.view(-1))
                    bandwidth = torch.sqrt(bandwidth_sq / 2.0) * self.bandwidth_factor
                    self.bandwidth = torch.clamp(bandwidth, self.min_bandwidth, self.max_bandwidth)

                kernel_matrix[i:end_i, j:end_j] = -dist_sq / ( 2 * self.bandwidth ** 2)

        return kernel_matrix

    def score_kernel(self, x, score_x, y, score_y):
        """
        Args:
            x: [B, D]
            score_x: [B, D]
            y: [B, D]
            score_y: [B, D]
        Returns:
            kernel_matrix: [B, B]
        """
        if self.training:
            self.update_bandwidth(x, y)
        
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # [B, B, D]
        dist_sq = torch.sum(diff ** 2, dim=-1)   # [B, B]
        kernel_mat = torch.exp(-dist_sq / (2 * self.bandwidth ** 2))  # [B, B]
        
        term1 = kernel_mat.unsqueeze(-1) * score_x.unsqueeze(1)  # [B, B, D]
        term1 = torch.sum(term1 * score_y.unsqueeze(0), dim=-1)  # [B, B]
        
        term2 = torch.sum(diff * score_y.unsqueeze(0), dim=-1)  # [B, B]
        term2 = term2 * kernel_mat / (self.bandwidth ** 2)
        
        term3 = torch.sum(diff ** 2, dim=-1) * kernel_mat / (self.bandwidth ** 4)
        term3 = term3 - kernel_mat * x.size(-1) / (self.bandwidth ** 2)
        
        return (term1 + term2 + term3)  # [B, B]

class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model,
            context_dim,
            n_head = 8,
            n_queries = 256,
            norm_layer = nn.LayerNorm,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)
        # self.proj = nn.Linear(d_model, d_model)
        self.pooler = AttentionPool2d(n_queries, d_model, num_heads=n_head, output_dim=d_model)
    def forward(self, x: torch.Tensor):
        N = x.shape[0]
        x = self.ln_k(x)
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(0).expand(N, -1, -1), x, x, need_weights=False)[0]
        out = self.pooler(out.permute(0,2,1))
        #   out = self.proj(out)
        return out[0]

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(1, spacial_dim + 1, embed_dim) / embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(0, 2, 1) # convert X shape (B, C, L) to (B, L, C)

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(x.shape[0], -1, -1) 
        x = torch.cat((self.cls_tokens, x), dim=1)
        x = x + self.positional_embedding[:, :, :].to(x.dtype)  # (L+1)NC
        x, att_map = self.mhsa(x[:, :1, :], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(1), att_map[:, :, 1:]



class SEGRA(torch.nn.Module):
    def __init__(self, network_config):
        super(SEGRA, self).__init__()
        
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        # ecg signal encoder
        self.ecg_model = network_config['ecg_model']
        self.num_leads = network_config['num_leads']

        if 'resnet' in self.ecg_model:
            if self.ecg_model == 'resnet18':
                model = ResNet18()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1, stride=3)
                self.att_pool_head = AttentionPool2d(spacial_dim=105,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet34':
                model = ResNet34()
                self.downconv = nn.Conv1d(in_channels=512, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet50':
                model = ResNet50()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            elif self.ecg_model == 'resnet101':
                model = ResNet101()
                self.downconv = nn.Conv1d(in_channels=2048, out_channels=self.proj_out, kernel_size=1)
                self.att_pool_head = AttentionPool2d(spacial_dim=313,
                                                    embed_dim=self.proj_out, 
                                                    num_heads=4, 
                                                    output_dim=self.proj_out)
            self.ratio = 0.1
            self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

        if 'vit' in self.ecg_model:
            if self.ecg_model == 'vit_tiny':
                model = vit_tiny(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_small':
                model = vit_small(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_middle':
                model = vit_middle(num_leads=self.num_leads)
            elif self.ecg_model == 'vit_base':
                model = vit_base(num_leads=self.num_leads)
            self.proj_e_input = model.width    
            self.proj_e = nn.Sequential(
                nn.Linear(self.proj_e_input, self.proj_hidden),
                nn.BatchNorm1d(self.proj_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.proj_hidden, self.proj_out),
                nn.BatchNorm1d(self.proj_out),
            )
            self.linear1 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)
            self.linear2 = nn.Linear(self.proj_e_input, self.proj_out, bias=False)


        self.ecg_encoder = model
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.query_attn = AttentionalPooler(self.proj_out, self.proj_out, n_queries=16)
        # text encoder
        url = network_config['text_model']
        self.lm_model = AutoModel.from_pretrained(
            url, trust_remote_code=True, revision='main')#, is_decoder=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            url, trust_remote_code=True, revision='main')
        
        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        
        self.image_score = nn.Sequential(
            nn.Linear(self.proj_out//2, self.proj_hidden),
            nn.LayerNorm(self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out//2)
        )
        
        self.text_score = nn.Sequential(
            nn.Linear(self.proj_out//2, self.proj_hidden),
            nn.LayerNorm(self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out//2)
        )

        self.stein_kernel = SteinKernel()
        # self.fuse_gate = nn.Sequential(
        #     nn.Linear(self.proj_out *2, self.proj_out),
        #     nn.Sigmoid()
        # )
        self.fuse_attn = nn.MultiheadAttention(embed_dim=self.proj_out, num_heads=4, batch_first=True)
        self.etm_head = nn.Linear(self.proj_out, 2)

    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output
    
    @torch.no_grad()
    def ext_ecg_emb(self, ecg):

        if 'resnet' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            ecg_emb = self.downconv(ecg_emb)
            ecg_token = ecg_emb.permute(0,2,1)
            proj_ecg_emb, att_map = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)

        if 'vit' in self.ecg_model:
            ecg_emb = self.ecg_encoder(ecg)
            proj_ecg_emb = self.proj_e(ecg_emb)

        query_emb = self.query_attn(ecg_token)
        proj_ecg_emb = proj_ecg_emb * F.sigmoid(query_emb)
        return proj_ecg_emb
    
    # @torch.no_grad()
    # def get_text_emb(self, input_ids, attention_mask):
    #     text_output = self.lm_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     last_hidden_state = text_output.hidden_states[-1]
    #     text_emb = last_hidden_state[:, -1]
    #     return text_emb#, last_hidden_state
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask)
        return text_emb.pooler_output#, text_emb.last_hidden_state

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def stein_alignment_loss(self, K1, K2, eps=1e-8):

        K1 = K1.clone()
        K2 = K2.clone()
        K1.fill_diagonal_(0)
        K2.fill_diagonal_(0)
        # K1 = center_kernel(K1)
        # K2 = center_kernel(K2)
        num = (K1 * K2).sum()
        denom = torch.sqrt((K1 * K1).sum() * (K2 * K2).sum() + eps)

        return 1 - (num / denom) ** 2
        # return torch.log1p(1 - (num / denom)**2)


    def stein_matrix(self, x, y, x_score, y_score):
        ecg_mu, ecg_logvar = torch.chunk(x, 2, dim=-1)
        text_mu, text_logvar = torch.chunk(y, 2, dim=-1)

        ecg_z = self.reparameterize(ecg_mu, ecg_logvar)
        text_z = self.reparameterize(text_mu, text_logvar)
        ecg_z = F.normalize(ecg_z, dim=-1, p=2)
        text_z = F.normalize(text_z, dim=-1, p=2)
        ecg_score = x_score(ecg_z)  # [B, L, D]
        text_score = y_score(text_z)     # [B, L, D]
        # ecg_score = F.normalize(ecg_score, dim=-1, p=2)
        # text_score = F.normalize(text_score, dim=-1, p=2)
        return ecg_z, ecg_score, text_z, text_score
    
    def stein_loss(self, ecg_z, ecg_score, text_z, text_score):
        #kernel_matrix = self.stein_kernel.score_kernel(ecg_z, ecg_score, text_z, text_score)    
        stein_ecg = self.stein_kernel.score_kernel(ecg_z, ecg_score, ecg_z, ecg_score)
        stein_text = self.stein_kernel.score_kernel(text_z, text_score, text_z, text_score)
        stein_cross = self.stein_kernel.score_kernel(ecg_z, ecg_score, text_z, text_score)

        alignment_loss =  self.stein_alignment_loss(stein_ecg, stein_text) #+ (loss_t + loss_i) / 2.0
        cross_loss = self.stein_alignment_loss(stein_ecg, stein_cross) \
            + self.stein_alignment_loss(stein_text, stein_cross)
        # kl_loss = self.compute_kl_loss(ecg_mu, ecg_logvar, text_mu, text_logvar)

        return (alignment_loss + cross_loss) # * 30  # + kl_loss

    def etm_loss(self, proj_ecg_emb, proj_text_emb):
        bs = proj_ecg_emb.shape[0]

        ecg_emb = []
        text_emb = []
        # positive pairs
        #etm_pos = torch.cat([proj_ecg_emb, proj_text_emb], dim=-1)
        # ecg_emb.append(proj_ecg_emb)
        # text_emb.append(proj_text_emb)

        # ECG hard negatives
        ecg_sim = F.softmax((proj_ecg_emb @ proj_ecg_emb.t()) / 0.07, dim=-1)
        ecg_sim.fill_diagonal_(0)
        ecg_neg_idx = torch.multinomial(ecg_sim, 1).squeeze(-1)
        neg_ecg_emb = proj_ecg_emb[ecg_neg_idx]
        ecg_emb.append(neg_ecg_emb)
        text_emb.append(proj_text_emb)
        # etm_neg_e = torch.cat([neg_ecg_emb, proj_text_emb], dim=-1)

        # Text hard negatives
        text_sim = F.softmax((proj_text_emb @ proj_text_emb.t()) / 0.07, dim=-1)
        text_sim.fill_diagonal_(0)
        text_neg_idx = torch.multinomial(text_sim, 1).squeeze(-1)
        neg_text_emb = proj_text_emb[text_neg_idx]
        ecg_emb.append(proj_ecg_emb)
        text_emb.append(neg_text_emb)
        # etm_neg_t = torch.cat([proj_ecg_emb, neg_text_emb], dim=-1)

        # Cross-modal hard negatives
        et_sim = F.softmax((proj_ecg_emb @ proj_text_emb.t()) / 0.07, dim=-1)
        et_sim.fill_diagonal_(0)
        et_neg_idx = torch.multinomial(et_sim, 1).squeeze(-1)
        neg_et_text_emb = proj_text_emb[et_neg_idx]
        ecg_emb.append(proj_ecg_emb)
        text_emb.append(neg_et_text_emb)
        # etm_neg_et = torch.cat([proj_ecg_emb, neg_et_text_emb], dim=-1)

        te_sim = F.softmax((proj_text_emb @ proj_ecg_emb.t()) / 0.07, dim=-1)
        te_sim.fill_diagonal_(0)
        te_neg_idx = torch.multinomial(te_sim, 1).squeeze(-1)
        neg_te_ecg_emb = proj_ecg_emb[te_neg_idx]
        ecg_emb.append(neg_te_ecg_emb)
        text_emb.append(proj_text_emb)
        # etm_neg_te = torch.cat([neg_te_ecg_emb, proj_text_emb], dim=-1)

        # concat all
        ecg_emb = torch.cat(ecg_emb, dim=0)
        text_emb = torch.cat(text_emb, dim=0)
        etm_emb_pos = torch.cat([proj_ecg_emb.unsqueeze(1), proj_text_emb.unsqueeze(1)], dim=1)
        #r = self.fuse_gate(etm_emb_pos)
        #etm_emb_pos = r * proj_ecg_emb + (1 - r) * proj_text_emb
        etm_emb_pos = self.fuse_attn(etm_emb_pos, etm_emb_pos, etm_emb_pos)[0].mean(dim=1)
        etm_output_pos = self.etm_head(etm_emb_pos)#.squeeze(-1)

        etm_emb_neg = torch.cat([ecg_emb.unsqueeze(1), text_emb.unsqueeze(1)], dim=1)
        #r_neg = self.fuse_gate(etm_emb_neg)
        #etm_emb_neg = r_neg * ecg_emb + (1 - r_neg) * text_emb
        etm_emb_neg = self.fuse_attn(etm_emb_neg, etm_emb_neg, etm_emb_neg)[0].mean(dim=1)
        etm_output_neg = self.etm_head(etm_emb_neg)#.squeeze(-1)

        return etm_output_pos, etm_output_neg



    def forward(self, ecg, input_ids, attention_mask):
        ecg_emb = self.ecg_encoder(ecg)

        if 'resnet' in self.ecg_model:
            # attention pooling (only for resnet models)
            ecg_emb = self.downconv(ecg_emb)
            ecg_token = ecg_emb.permute(0, 2, 1)  # (B, L, C)
            proj_ecg_emb, _ = self.att_pool_head(ecg_emb)
            proj_ecg_emb = proj_ecg_emb.view(proj_ecg_emb.shape[0], -1)
            ecg_emb = self.avgpool(ecg_emb).view(ecg_emb.shape[0], -1)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))           
            # ecg_emb = torch.mean(ecg_emb, dim=-1)
        
        if 'vit' in self.ecg_model:
            proj_ecg_emb = self.proj_e(ecg_emb)
            ecg_emb1 = self.dropout1(self.linear1(ecg_emb))
            ecg_emb2 = self.dropout2(self.linear2(ecg_emb))

        query_emb = self.query_attn(ecg_token)
        proj_ecg_emb = proj_ecg_emb * F.sigmoid(query_emb)
        proj_ecg_emb = normalize(proj_ecg_emb, dim=-1)


        # get text feature
        # text feature extraction is independent of the type of ecg encoder
        text_emb = self.get_text_emb(input_ids, attention_mask)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)


        ecg_z, ecg_score, text_z, text_score = self.stein_matrix(proj_ecg_emb, proj_text_emb,
                                                                self.image_score, self.text_score)
        
        etm_output_pos, etm_output_neg = self.etm_loss(proj_ecg_emb, proj_text_emb)
        # ecg_z1, ecg_score1, ecg_z2, ecg_score2 = self.stein_matrix(ecg_emb1, ecg_emb2,
        #                                                         self.image_score, self.image_score)
        if self.training:
            return {'ecg_emb': [ecg_emb1, ecg_emb2],
                    'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb],
                    'stein_input': [ecg_z, ecg_score, text_z, text_score],
                    'etm_output': [etm_output_pos, etm_output_neg],
                    }
        else:
            return {'ecg_emb': [ecg_emb1, ecg_emb2],
                    'proj_ecg_emb': [proj_ecg_emb],
                    'proj_text_emb': [proj_text_emb],
                    'stein_input': [ecg_z, ecg_score, text_z, text_score],
                    'etm_output': [etm_output_pos, etm_output_neg],
                    }
