import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed_pos import PatchEmbedding_C_group
import copy

class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)   
        self.linear = nn.Linear(nf, target_window) 
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  
        x = self.flatten(x)          
        x = self.linear(x)            
        x = self.dropout(x)
        return x

class DiffusionNoise(nn.Module):
    """
    Diffusion 前向过程的单步扰动：x_noisy = x + noise_scale * epsilon, where epsilon ~ N(0, I)
    """
    def __init__(self, noise_scale=0.1):
        super().__init__()
        self.noise_scale = noise_scale

    def forward(self, x):
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(x) * self.noise_scale
            return x + noise
        return x

class MultiScaleConv(nn.Module):
    """轻量并行多尺度卷积"""
    def __init__(self, d_model, kernel_sizes=(3,7,15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k, padding=k//2, bias=True)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(len(kernel_sizes)*d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)     
        outs = [conv(x) for conv in self.convs]   
        out = torch.cat(outs, dim=1)              
        out = out.transpose(1, 2)                  
        out = self.proj(out)                      
        return self.act(out)                      

class CrossBlock(nn.Module):
    """
    结构: CrossAttn -> Add&Norm -> FeedForward -> Add&Norm
    """
    def __init__(self, d_model, d_ff, n_heads, dropout=0.1, activation="relu", factor=10):
        super().__init__()
        self.cross_attention = AttentionLayer(
            FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Feed Forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

    def forward(self, x, cross):
        # x is Query (Long Window), cross is Key/Value (Short Window)
        # Cross Attention
        out, _ = self.cross_attention(
            x, cross, cross,
            attn_mask=None
        )
        x = x + self.dropout(out)
        x = self.norm1(x)

        # Feed Forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)
    

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name

        # 确保从大到小排序，从而保证从大到小 cross attention
        self.seq_len_list = sorted(configs.seq_len_list, reverse=True) 
        
        self.pred_len = configs.pred_len
        
        self.alpha = torch.nn.Parameter(torch.tensor(configs.tau_hat_init)) 
        

        padding = configs.stride
        patch_len = configs.patch_len
        stride = configs.stride

        noise_scale = configs.noise_scale
        self.diffusion_noise = DiffusionNoise(noise_scale=noise_scale)

        self.configs = configs

        # feature groups
        if configs.data_path == '/data/stock_daily_2005_2021.feather':
            self.feature_group = [[0],[1,2,3,4,5],[6],[7],[8]] 
        else:
            self.feature_group = [[0],[1],[2,3,4,5,6],[7],[8]] 

        self.total_d_model = len(self.feature_group) * configs.d_model

        self.patch_embedding = PatchEmbedding_C_group(
            configs.d_model, patch_len, stride, padding, configs.dropout, self.feature_group
        )

        # 多尺度卷积
        self.multi_scale = MultiScaleConv(configs.d_model, kernel_sizes=(3,7,15))

        #cross attetion
        self.cross_layers = nn.ModuleList()
        if len(self.seq_len_list) > 1:
            for _ in range(len(self.seq_len_list) - 1):
                self.cross_layers.append(
                    CrossBlock(
                        d_model=self.total_d_model,
                        d_ff=configs.d_ff * len(self.feature_group),
                        n_heads=configs.n_heads,
                        dropout=configs.dropout,
                        activation=configs.activation,
                        factor=configs.factor
                    )
                )

        # Encoder，各个窗口Self-Attention
        self.global_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor,
                                      attention_dropout=configs.dropout,
                                      output_attention=False),
                        self.total_d_model,
                        configs.n_heads
                    ),
                    self.total_d_model,
                    configs.d_ff * len(self.feature_group),
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.total_d_model)
        )
                

      

        max_seq_len = max(self.seq_len_list)
        self.max_head_nf = int((max_seq_len - patch_len) / stride + 2)

        if self.task_name in ('multiple_regression','predict_feature'):
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.max_head_nf * self.total_d_model, configs.output_channels)

            
        self.head_nf_total = 0
        for sl in self.seq_len_list:
            num_patches = int((sl - patch_len) / stride + 2)
            self.head_nf_total += num_patches

        if self.task_name in ('Long_term_forecasting', 'short_term_forecast'):
            self.head = FlattenHead(configs.enc_in, configs.d_model * self.head_nf * len(self.feature_group), configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name in ('imputation', 'anomaly_detection'):
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len, head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)
        


    def _encode_single_window_conv(self, x_enc, start_index=0):
        """
        仅执行 Embedding 和 Conv
        """
        B = x_enc.shape[0]
        x_in = x_enc.permute(0, 2, 1) # [B, Vars, L]

        # Patch Embedding -> List of [B, Patch, D] (per group)
        enc_out_list, n_vars = self.patch_embedding(x_in, start_index=start_index)
        
        # 将所有 Group 拼在一起进行 Conv 处理 (Batch 维度堆叠)
        # sizes: list of batch sizes equivalent for split later? No, shape[0] is sumG*B
        split_sizes = [t.shape[0] for t in enc_out_list]
        enc_inputs = torch.cat(enc_out_list, dim=0)   # (sumG*B, P, D)

        # Apply Multi-scale Conv separately per patch/group
        enc_inputs = self.multi_scale(enc_inputs)     # (sumG*B, P, D)


        # Split back to groups and concatenate channel-wise
        splits = torch.split(enc_inputs, split_sizes, dim=0)
        
        group_tensors = []
        for s in splits:         
            patch_num = s.shape[1]
            d_model = s.shape[2]
            s = s.contiguous().view(B, 1, patch_num, d_model)
            s = s.permute(0, 1, 3, 2) # [B, 1, D, Patch]
            group_tensors.append(s)

        out_cat = torch.cat(group_tensors, dim=2) 
        return out_cat, n_vars
    
    def regression(self, x_enc_list):
        features_with_len = []
        
        max_seq_len_global = self.seq_len_list[0] # 大到小排序后的第一个
        max_patch_num = int((max_seq_len_global - self.configs.patch_len) / self.configs.stride + 2)

        for x_enc in x_enc_list:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev
            
            curr_len = x.shape[1]
            curr_patch_num = int((curr_len - self.configs.patch_len) / self.configs.stride + 2)
            
            # 计算 offset 以对齐位置编码 (右对齐/末尾对齐)
            offset = max_patch_num - curr_patch_num
            
            #  Embed -> Conv
            out_cat, _ = self._encode_single_window_conv(x, start_index=offset)
            
            # out_cat: [B, 1, Total_D, Patch] -> [B, Patch, Total_D]
            out_formatted = out_cat.squeeze(1).permute(0, 2, 1)
            
            features_with_len.append((curr_len, out_formatted))

        # 强制按照序列长度从大到小排序
        features_with_len.sort(key=lambda x: x[0], reverse=True)
        
        # 提取排序后的特征
        sorted_features = [f[1] for f in features_with_len]

        # Cross Attention Cascade
        # Q = Long Window, KV = Short Window
        curr_query = sorted_features[0] # 最长窗口作为初始 Query

        if len(sorted_features) > 1:
            for i in range(1, len(sorted_features)):
                curr_kv = sorted_features[i] # 当前较短窗口

                #diffusion
                curr_kv = self.diffusion_noise(curr_kv)
                
                # Cross Attention: Q(Long) query KV(Short)
                # 结果长度保持 Q 的长度
                curr_query = self.cross_layers[i-1](x=curr_query, cross=curr_kv)

        #  Global Self-Attention
        # 对融合了所有窗口信息的长序列进行 Self-Attention
        final_out, _ = self.global_encoder(curr_query)
        
       
        output = self.flatten(final_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)

        return output

    def forward(self, x_enc_list, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in ('Long_term_forecasting', 'short_term_forecast'):
            dec_out = self.forecast(x_enc_list, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :] 
        if self.task_name == 'imputation':
            return self.imputation(x_enc_list, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc_list)
        if self.task_name == 'classification':
            return self.classification(x_enc_list, x_mark_enc)
        if self.task_name == 'multiple_regression':
            return self.regression(x_enc_list)
        return None






