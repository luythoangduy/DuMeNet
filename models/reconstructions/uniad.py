import copy
import math
import os
import random
from typing import Optional
import json
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

__all__ = ["UniADMemory"]


class UniADMemory(nn.Module):
    def __init__(
        self,
        inplanes,
        instrides,
        feature_size,
        feature_jitter,
        neighbor_mask,
        hidden_dim,
        pos_embed_type,
        save_recon,
        initializer,
        stats_config,
        **kwargs,
    ):
        super().__init__()
        assert isinstance(inplanes, list) and len(inplanes) == 1
        assert isinstance(instrides, list) and len(instrides) == 1
        self.feature_size = feature_size
        self.num_queries = feature_size[0] * feature_size[1]
        self.feature_jitter = feature_jitter
        self.feature_masking = kwargs.get('feature_masking', None)  # Dict với 'ratio' và 'prob'
        self.pos_embed = build_position_embedding(
            pos_embed_type, feature_size, hidden_dim
        )
        self.save_recon = save_recon
        self.input_channel_dim = inplanes[0]
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            kwargs.get('nhead', 8), 
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False)
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if kwargs.get('normalize_before', False) else None
        self.encoder = TransformerEncoder(
            encoder_layer, 
            kwargs.get('num_encoder_layers', 4),
            encoder_norm
        )
        
        # Decoder
        decoder_layer = TransformerMemoryDecoderLayer(
            hidden_dim,
            kwargs.get('nhead', 8),
            kwargs.get('dim_feedforward', 1024),
            kwargs.get('dropout', 0.1),
            kwargs.get('activation', 'relu'),
            kwargs.get('normalize_before', False),
        )
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            kwargs.get('num_decoder_layers', 4),
            decoder_norm,
            return_intermediate=False,
        )
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])
        self.stats_config = stats_config # Lưu config
        self.channel_k_values = self.load_channel_k_values()
        
        # Upsampling
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=instrides[0])

        # Initialize parameters
        initialize_from_cfg(self, initializer)

    def add_jitter(self, feature_tokens, scale, prob):
        if random.uniform(0, 1) <= prob:
            num_tokens, batch_size, dim_channel = feature_tokens.shape
            feature_norms = (
                feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel
            )  # (H x W) x B x 1
            jitter = torch.randn((num_tokens, batch_size, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens
    def compute_stats(self, tensor: torch.Tensor, name: str) -> dict:
        """Tính toán min, max, mean, std cho một tensor."""
        stats = {
            f'{name}_min': tensor.min().item(),
            f'{name}_max': tensor.max().item(),
            f'{name}_mean': tensor.mean().item(),
            f'{name}_std': tensor.std().item(),
        }
        # print(f"{name} stats: min={stats[f'{name}_min']}, max={stats[f'{name}_max']}, mean={stats[f'{name}_mean']}, std={stats[f'{name}_std']}")
        return stats
    
    def load_channel_k_values(self):
        """Loads channel-wise k_ci_value from the combined stats file."""
        stats_file_path = self.stats_config.get('stats_file', 'analysis_results/feature_stats_combined.json')
        ci_ratio = self.stats_config.get('ci_ratio', 80) # Default là 80%
        
        if not os.path.exists(stats_file_path):
            print(f"WARNING: Stats file not found at {stats_file_path}. Using default k=1.0 for all channels.")
            # Fallback to default k=1.0 if file not found
            return torch.ones(self.input_channel_dim, dtype=torch.float32)

        try:
            with open(stats_file_path, 'r') as f:
                stats = json.load(f)
        except Exception as e:
            print(f"ERROR reading stats file {stats_file_path}: {e}. Using default k=1.0.")
            return torch.ones(self.input_channel_dim, dtype=torch.float32)

        k_values = []
        ci_key = f"{ci_ratio}%_CI"
        
        # Lấy số kênh (C) từ shape đầu tiên (Feature Align)
        num_channels = stats['global_stats']['feature_shape'][1] 
        
        # Đảm bảo danh sách channel_stats có đủ kênh
        if len(stats['channel_stats']) != num_channels:
            print(f"WARNING: Expected {num_channels} channels, found {len(stats['channel_stats'])}. Using default k=1.0.")
            return torch.ones(self.input_channel_dim, dtype=torch.float32)

        for channel_stat in stats['channel_stats']:
            try:
                # Trích xuất k_ci_value tương ứng
                k_val = channel_stat['percentiles'][ci_key]['k_ci_value']
                k_values.append(k_val)
            except KeyError:
                print(f"WARNING: k_ci_value for {ci_key} not found in channel {channel_stat['channel_idx']}. Using default k=1.0.")
                k_values.append(1.0)
                
        # Chuyển list sang tensor và đặt vào device (sẽ được move cùng module sau)
        k_tensor = torch.tensor(k_values, dtype=torch.float32)
        
        # Kích thước phải là [C]
        if k_tensor.shape[0] != self.input_channel_dim:
             # Nếu hidden_dim != C, cần phải kiểm tra lại (thường hidden_dim = C ở lớp này)
             print(f"ERROR: Loaded K dimension {k_tensor.shape[0]} != hidden_dim {self.hidden_dim}. Using default k=1.0.")
             return torch.ones(self.input_channel_dim, dtype=torch.float32)

        print(f"Successfully loaded {len(k_values)} channel K-values for {ci_key}.")
        return nn.Parameter(k_tensor, requires_grad=False) # Lưu K dưới dạng Parameter không cần gradient

    
    def forward(self, input):
        feature_align = input["feature_align"]  # B x C X H x W (B x 272 x 14 x 14)
        # feature_align = feature_align + 0.058038
        backbone_output_stats = self.compute_stats(feature_align, "backbone_output")
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        # Add jitter during training if enabled
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        
        # Project input features
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C_hidden (196 x B x 256)
        
        # SỬA DÒNG NÀY: Lấy K values và căn chỉnh kích thước cho phép nhân/broadcast
        k_channel_values = self.channel_k_values.to(feature_align.device)

        # 1. Kích thước cho feature_rec_tokens (H*W x B x C_output): cần (1, 1, C)
        k_token_aligned = k_channel_values.unsqueeze(0).unsqueeze(0) 
        
        # 2. Kích thước cho feature_align (B x C x H x W): cần (1, C, 1, 1)
        k_spatial_aligned = k_channel_values.view(1, -1, 1, 1)
        
        # k_channel = self.channel_k_values.unsqueeze(0).unsqueeze(0) # DÒNG CŨ
        # k = 0.57
        feature_tokens = F.layer_norm(feature_tokens, feature_tokens.shape[-1:])
        
        # Get positional embeddings
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        
        # Encode features using transformer encoder
        encoded_tokens = self.encoder(
            feature_tokens, pos=pos_embed
        )  # (H x W) x B x C
        encoded_stats = self.compute_stats(encoded_tokens, "encoded_feature")

        # Decode features
        decoded_tokens = self.decoder(
            encoded_tokens, 
            encoded_tokens, 
            pos=pos_embed
        )  # (H x W) x B x C
        # Project back to original dimension
        feature_rec_tokens = self.output_proj(decoded_tokens)  # (H x W) x B x C_output
        decoder_output_stats = self.compute_stats(feature_rec_tokens, "decoder_output_raw")
        
        # SỬA DÒNG NÀY: Nhân k_token_aligned (1 x 1 x 272) với feature_rec_tokens (H*W x B x 272)
        feature_rec_tokens = torch.sigmoid(feature_rec_tokens * k_token_aligned) 
        decoder_tokens_sigmoid_stats = self.compute_stats(feature_rec_tokens, "decoder_output_sigmoid")

        # Reshape back to spatial representation
        feature_rec = rearrange(
            feature_rec_tokens, "(h w) b c -> b c h w", h=self.feature_size[0]
        )  # B x C X H x W

        # Save reconstructed features if needed
        if not self.training and self.save_recon:
            clsnames = input["clsname"]
            filenames = input["filename"]
            for clsname, filename, feat_rec in zip(clsnames, filenames, feature_rec):
                filedir, filename = os.path.split(filename)
                _, defename = os.path.split(filedir)
                filename_, _ = os.path.splitext(filename)
                save_dir = os.path.join(self.save_recon.save_dir, clsname, defename)
                os.makedirs(save_dir, exist_ok=True)
                feature_rec_np = feat_rec.detach().cpu().numpy()
                np.save(os.path.join(save_dir, filename_ + ".npy"), feature_rec_np)

        # Compute prediction (reconstruction error)
        feature_align = torch.sigmoid(feature_align * k_spatial_aligned) 
        feature_align_sigmoid_stats = self.compute_stats(feature_align, "feature_align_sigmoid")
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        
        pred = self.upsample(pred)  # B x 1 x H x W
        
        # Prepare output dictionary based on available memories
        output_dict = {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
            "backbone_output_stats": backbone_output_stats,
            "encoded_feature_stats": encoded_stats,
            "decoder_output_raw_stats": decoder_output_stats,
            "decoder_output_sigmoid_stats": decoder_tokens_sigmoid_stats,
            "feature_align_sigmoid_stats": feature_align_sigmoid_stats,
        }
        
        return output_dict


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src
        pos = torch.cat(
            [pos.unsqueeze(1)] * src.size(1), dim=1
        )  # (H X W) x B x C

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = tgt
        pos = torch.cat(
            [pos.unsqueeze(1)] * tgt.size(1), dim=1
        )  # (H X W) x B x C

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

class EfficientMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Placeholder for Linear/Efficient Attention logic (ví dụ: Softmax cho Q, K)
        # Thực tế, bạn sẽ thay thế bằng logic Linear Attention cụ thể của mình
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        T, B, C = query.shape # Token length, Batch size, Channels
        
        # 1. Project QKV
        qkv = self.qkv_proj(query) # T x B x 3C
        qkv = qkv.reshape(T, B, 3, self.num_heads, self.head_dim).permute(2, 1, 3, 0, 4) # 3 x B x H x T x D_h
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 2. Efficient Attention Core (ví dụ: Kernel-based/Softmax-on-Q-K)
        # Ví dụ: Softmax trên Q và K (như trong một số biến thể Linear Attention)
        q = F.softmax(q * self.scaling, dim=-1) # B x H x T x D_h
        k = F.softmax(k * self.scaling, dim=-2) # B x H x T x D_h (Softmax trên chiều T)
        
        # Linear Attention: Q * (K^T * V)
        kv = torch.einsum("bhsd,bhse->bhde", k, v) # B x H x D_h x D_h
        attn_output = torch.einsum("bhsd,bhde->bhse", q, kv) # B x H x T x D_h
        
        # 3. Reshape và Output Projection
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(T, B, C) # T x B x C
        attn_output = self.dropout(self.out_proj(attn_output))
        
        # Trả về output và None (như nn.MultiheadAttention)
        return attn_output, None
    
class TransformerMemoryDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Standard transformer decoder layer components
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # self.self_attn = EfficientMultiheadAttention(hidden_dim, nhead, dropout=dropout)
        # self.multihead_attn = EfficientMultiheadAttention(hidden_dim, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        # Self attention
        q = k = self.with_pos_embed(tgt, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "celu":
        return F.celu
    if activation == "selu":
        return F.selu
    if activation == "silu":
        return F.silu
    if activation == "elu":
        return F.elu
    raise RuntimeError


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]), device=tensor.device)  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2
        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        return pos

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    if pos_embed_type in ("v2", "sine"):
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed