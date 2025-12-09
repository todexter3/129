import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer, DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_C_group
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


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name

        # 确保从大到小排序，从而保证从大到小 cross attention
        self.seq_len_list = sorted(configs.seq_len_list, reverse=True) 
        
        self.pred_len = configs.pred_len
        
        self.alpha = torch.nn.Parameter(torch.tensor(configs.tau_hat_init))  # 可学习的tau_hat参数
        

        padding = configs.stride
        patch_len = configs.patch_len
        stride = configs.stride
        self.configs = configs

        # feature groups
        if configs.data_path == '/data/stock_daily_2005_2021.feather':
            self.feature_group = [[0],[1,2,3,4,5],[6],[7],[8]] 
        else:
            self.feature_group = [[0],[1],[2,3,4,5,6],[7],[8]] 

        # 特征拼接后的总维度 groups * d_model
        self.total_d_model = len(self.feature_group) * configs.d_model

        # Patch Embedding
        self.patch_embedding = PatchEmbedding_C_group(
            configs.d_model, patch_len, stride, padding, configs.dropout, self.feature_group
        )

        # Encoder，各个窗口Self-Attention，e_layers
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )

        # 多尺度卷积
        self.multi_scale = MultiScaleConv(configs.d_model, kernel_sizes=(3,7,15))

        # Cross-Attention, N-1 个 DecoderLayer 来融合 N 个窗口
        self.cascade_decoders = nn.ModuleList()
        if len(self.seq_len_list) > 1:
            for _ in range(len(self.seq_len_list) - 1):
                self.cascade_decoders.append(
                    DecoderLayer(
                        # Self-Attention融合前微调当前 Query (Short Window)
                        self_attention=AttentionLayer(
                            FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            self.total_d_model, configs.n_heads
                        ),
                        # Cross-Attention: Q = Short Window, K/V = Long Window Context
                        cross_attention=AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                            self.total_d_model, configs.n_heads
                        ),
                        d_model=self.total_d_model,
                        d_ff=configs.d_ff * len(self.feature_group), 
                        dropout=configs.dropout,
                        activation=configs.activation
                    )
                )

        # Prediction Head 
        min_seq_len = self.seq_len_list[-1]
        self.min_head_nf = int((min_seq_len - patch_len) / stride + 2)

        if self.task_name in ('multiple_regression','predict_feature'):
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.min_head_nf * self.total_d_model, configs.output_channels)
            
        # 兼容其他 task (虽然逻辑可能需要对应修改)

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
        

    def _encode_all_groups_once(self, x_enc):
        B = x_enc.shape[0]
        x_in = x_enc.permute(0, 2, 1) 
        
        # Patch Embedding
        enc_out_list, n_vars = self.patch_embedding(x_in)
        
        # Encoder
        split_sizes = [t.shape[0] for t in enc_out_list]  
        enc_inputs = torch.cat(enc_out_list, dim=0)      
        enc_outputs, attns = self.encoder(enc_inputs)    
        
        # conv
        enc_outputs = self.multi_scale(enc_outputs)      
        
        splits = torch.split(enc_outputs, split_sizes, dim=0)  
        
        group_tensors = []
        for s in splits:
            patch_num = s.shape[1]
            d_model = s.shape[2]
            s = s.contiguous().view(B, 1, patch_num, d_model)
            s = s.permute(0, 1, 3, 2) 
            group_tensors.append(s)

        out_cat = torch.cat(group_tensors, dim=2) 
        return out_cat, n_vars

    def regression(self, x_enc_list):
        # 对每个窗口独立进行 Self-Attention
        encoded_results = []
        
        for x_enc in x_enc_list:
            # Normalization
            means = x_enc.mean(1, keepdim=True).detach()
            x = x_enc - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

            # encoder 
            out_cat, n_vars = self._encode_all_groups_once(x)  
            
            out_formatted = out_cat.squeeze(1).permute(0, 2, 1) 
            
            encoded_results.append(out_formatted)

        encoded_results.sort(key=lambda x: x.shape[1], reverse=True)

        #  Cross-Attention
        # 初始 KV为最长窗口的特征
        curr_kv = encoded_results[0] 

        if len(encoded_results) > 1:
            for i in range(1, len(encoded_results)):
                # Q 为下一个较短窗口的特征
                curr_query = encoded_results[i]

                curr_kv = self.cascade_decoders[i-1](x=curr_query, cross=curr_kv)
                
      
        final_out = curr_kv 

        # Projection 
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

