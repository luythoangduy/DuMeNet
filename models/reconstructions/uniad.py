import copy
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from models.initializer import initialize_from_cfg
from torch import Tensor, nn

__all__ = ["UniADMemory"]


class ChannelMemoryModule(nn.Module):
    """
    Channel Memory Module - Feature-wise processing với Q/K/V projections
    Memory shape: [mem_size, feature_dim] - học patterns trong feature space
    """
    def __init__(self, mem_dim, feature_dim, **kwargs):
        super(ChannelMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.feature_dim = feature_dim  # Feature dimension
        self.scale = 1.0 / math.sqrt(feature_dim)  # Scale factor for attention
        
        # Memory bank: [mem_size, feature_dim]
        self.memory = nn.Parameter(torch.randn(mem_dim, feature_dim))
        nn.init.normal_(self.memory, mean=0, std=0.1)
        
        # Q, K, V projections như trong attention mechanism
        self.query_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.key_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        self.value_proj = nn.Linear(feature_dim, feature_dim, bias=False)
        
    def forward(self, input_tokens):
        """
        Args:
            input_tokens: [N_tokens, batch_size, feature_dim] - feature tokens
        Returns:
            dict with output and memory information
        """
        N_tokens, batch_size, feature_dim = input_tokens.shape
        
        # Reshape for processing: [N_tokens * batch_size, feature_dim]
        input_flat = input_tokens.view(N_tokens * batch_size, feature_dim)
        
        # Project input to queries
        queries = self.query_proj(input_flat)  # [N_tokens * batch_size, feature_dim]
        
        # Project memory to keys and values
        keys = self.key_proj(self.memory)  # [mem_dim, feature_dim]
        values = self.value_proj(self.memory)  # [mem_dim, feature_dim]
        
        # Compute attention scores: Q @ K^T
        attention_scores = torch.mm(queries, keys.t())  # [N_tokens * batch_size, mem_dim]

        if self.training: # Mask memory slot 20%
            mask_ratio = 0.2
            num_masked = int(self.mem_dim * mask_ratio)
            if num_masked > 0:
                # Chọn ngẫu nhiên index để mask
                mask_indices = torch.randperm(self.mem_dim, device=attention_scores.device)[:num_masked]
                # Gán -inf để Softmax bỏ qua các slot này (coi như không tồn tại khi query)
                attention_scores[:, mask_indices] = float('-inf')
        
        # Apply scale
        attention_scores = attention_scores * self.scale
        
        # Apply softmax to get attention weights
        att_weight = F.softmax(attention_scores, dim=1)  # [N_tokens * batch_size, mem_dim]
        
        # Retrieve from memory: attention × values
        output_flat = torch.mm(att_weight, values)  # [N_tokens * batch_size, feature_dim]
        
        # Reshape back to original format
        output_tokens = output_flat.view(N_tokens, batch_size, feature_dim)  # [N_tokens, batch_size, feature_dim]
        
        return {
            'output': output_tokens,
            'att_weight': att_weight,
            'attention_scores': attention_scores,
            'memory': self.memory
        }


class SpatialMemoryModule(nn.Module):
    """
    Spatial Memory Module - Spatial pattern processing với Q/K/V projections và SSIM similarity
    Memory shape: [mem_size, H, W] - học spatial patterns
    """
    def __init__(self, mem_dim, height, width, **kwargs):
        super(SpatialMemoryModule, self).__init__()
        
        self.mem_dim = mem_dim  # Number of memory slots
        self.height = height    # Spatial height
        self.width = width      # Spatial width
        self.spatial_dim = height * width
        self.scale = 1.0  # Scale factor for SSIM similarity
        
        # Memory bank: [mem_size, H, W] - giữ nguyên spatial structure
        self.memory = nn.Parameter(torch.randn(mem_dim, height, width))
        nn.init.normal_(self.memory, mean=0, std=0.1)
        
        # Q, K, V projections cho spatial patterns
        self.query_proj = nn.Linear(self.spatial_dim, self.spatial_dim, bias=False)
        self.key_proj = nn.Linear(self.spatial_dim, self.spatial_dim, bias=False)
        self.value_proj = nn.Linear(self.spatial_dim, self.spatial_dim, bias=False)

    def compute_ssim_similarity(self, query_patterns, memory_patterns):
        """
        Compute SSIM similarity between query spatial patterns and memory patterns
        
        Args:
            query_patterns: [N_patterns, H, W] - query spatial patterns
            memory_patterns: [mem_dim, H, W] - memory spatial patterns
        Returns:
            similarity: [N_patterns, mem_dim] - SSIM similarities
        """
        N_patterns, H, W = query_patterns.shape
        mem_dim = memory_patterns.shape[0]
        
        # Flatten spatial dimensions
        query_flat = query_patterns.view(N_patterns, H * W)  # [N_patterns, H*W]
        memory_flat = memory_patterns.view(mem_dim, H * W)  # [mem_dim, H*W]
        
        # Compute means
        query_mean = torch.mean(query_flat, dim=1, keepdim=True)  # [N_patterns, 1]
        memory_mean = torch.mean(memory_flat, dim=1, keepdim=True)  # [mem_dim, 1]
        
        # Compute variances
        query_var = torch.var(query_flat, dim=1, keepdim=True)  # [N_patterns, 1]
        memory_var = torch.var(memory_flat, dim=1, keepdim=True)  # [mem_dim, 1]
        
        # Center the data
        query_centered = query_flat - query_mean
        memory_centered = memory_flat - memory_mean
        
        # Compute covariance
        covariance = torch.mm(query_centered, memory_centered.t()) / (H * W - 1)  # [N_patterns, mem_dim]
        
        # SSIM formula components
        c1, c2 = 0.01, 0.03
        
        # Numerator: (2*mu1*mu2 + c1) * (2*cov + c2)
        mean_product = torch.mm(query_mean, memory_mean.t())  # [N_patterns, mem_dim]
        numerator = (2 * mean_product + c1) * (2 * covariance + c2)
        
        # Denominator: (mu1^2 + mu2^2 + c1) * (var1 + var2 + c2)
        mean_sum = query_mean**2 + memory_mean.t()**2  # [N_patterns, mem_dim]
        var_sum = query_var + memory_var.t()  # [N_patterns, mem_dim]
        denominator = (mean_sum + c1) * (var_sum + c2)
        
        # SSIM similarity
        ssim = numerator / (denominator + 1e-8)
        
        return ssim

    def forward(self, input_tokens):
        """
        Args:
            input_tokens: [N_tokens, batch_size, feature_dim] - feature tokens
        Returns:
            dict with output and memory information
        """
        feature_dim, batch_size, N_tokens = input_tokens.shape
        
        # Reshape để có spatial dimensions (giả sử feature_dim = height * width)
        H, W = self.height, self.width
        
        # Reshape to spatial format: [N_tokens * batch_size, H, W]
        input_spatial = input_tokens.view(N_tokens * batch_size, H, W)
        
        # Project input to queries
        input_flat = input_spatial.view(N_tokens * batch_size, H * W)  # [N_tokens * batch_size, H*W]
        queries_flat = self.query_proj(input_flat)  # [N_tokens * batch_size, H*W]
        queries_spatial = queries_flat.view(N_tokens * batch_size, H, W)  # [N_tokens * batch_size, H, W]
        
        # Project memory to keys and values
        memory_flat = self.memory.view(self.mem_dim, H * W)  # [mem_dim, H*W]
        keys_flat = self.key_proj(memory_flat)  # [mem_dim, H*W]
        values_flat = self.value_proj(memory_flat)  # [mem_dim, H*W]
        keys_spatial = keys_flat.view(self.mem_dim, H, W)  # [mem_dim, H, W]
        
        # Compute SSIM similarity between queries và keys
        ssim_similarity = self.compute_ssim_similarity(queries_spatial, keys_spatial)  # [N_tokens * batch_size, mem_dim]

        if self.training: # Mask memory slot 20%
            mask_ratio = 0.2
            num_masked = int(self.mem_dim * mask_ratio)
            if num_masked > 0:
                # Chọn ngẫu nhiên index để mask
                mask_indices = torch.randperm(self.mem_dim, device=ssim_similarity.device)[:num_masked]
                # Gán -inf vào similarity
                ssim_similarity[:, mask_indices] = float('-inf')
        
        # Apply scale and softmax to get attention weights
        attention_scores = ssim_similarity * self.scale
        att_weight = F.softmax(attention_scores, dim=1)  # [N_tokens * batch_size, mem_dim]
        
        # Retrieve from memory: attention × values
        output_flat = torch.mm(att_weight, values_flat)  # [N_tokens * batch_size, H*W]
        
        # Reshape back to original token format
        output_tokens = output_flat.view(N_tokens, batch_size, feature_dim)  # [N_tokens, batch_size, feature_dim]
        
        return {
            'output': output_tokens,
            'att_weight': att_weight,
            'ssim_similarity': ssim_similarity,
            'memory': self.memory
        }

class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim,
        feature_size,
        neighbor_mask,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.neighbor_mask = neighbor_mask

        # Sử dụng các lớp Encoder/Decoder layer hiện có
        encoder_layer = TransformerEncoderLayer( 
            hidden_dim, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(hidden_dim) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        decoder_layer = TransformerMemoryDecoderLayer( # DÙNG LỚP DECODER MỚI
            hidden_dim,
            nhead, # feature_size đã bị bỏ khỏi TransformerMemoryDecoderLayer
            dim_feedforward,
            dropout,
            activation,
            normalize_before,
        ) 
        decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
        )

        self.hidden_dim = hidden_dim
        self.nhead = nhead

    def generate_mask(self, feature_size, neighbor_size):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size
        hm, wm = neighbor_size
        mask = torch.ones(h, w, h, w)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                    idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
            .to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        ) # Đảm bảo mask cùng device
        return mask

    def forward(self, src, pos_embed):
        _, batch_size, _ = src.shape
        pos_embed_batch = torch.cat(
            [pos_embed.unsqueeze(1)] * batch_size, dim=1
        )  # (H X W) x B x C

        if self.neighbor_mask:
            mask = self.generate_mask(
                self.feature_size, self.neighbor_mask.neighbor_size
            )
            mask_enc = mask if self.neighbor_mask.mask[0] else None
            mask_dec1 = mask if self.neighbor_mask.mask[1] else None # tgt_mask
            mask_dec2 = mask if self.neighbor_mask.mask[2] else None # memory_mask
        else:
            mask_enc = mask_dec1 = mask_dec2 = None

        output_encoder = self.encoder(
            src, mask=mask_enc, pos=pos_embed_batch
        )  # (H X W) x B x C
        # output_decoder nhận tgt là features sau khi retrieval/fusion, 
        # và memory là encoded_tokens.
        # Tuy nhiên, kiến trúc ban đầu của UniAD là output_decoder = self.decoder(output_encoder)
        # Để giữ kiến trúc Memory hiện tại, ta KHÔNG DÙNG FORWARD CỦA LỚP TRANSFORMER NÀY

        # Để phù hợp với UniADMemory, bạn chỉ cần output_encoder, phần mask sẽ được tạo ở đây
        # và áp dụng trong lớp Encoder/Decoder
        return output_encoder, mask_enc, mask_dec1, mask_dec2, pos_embed_batch

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
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(inplanes[0], hidden_dim)
        
        # Memory modules configuration
        self.memory_mode = kwargs.get('memory_mode', 'both')  # 'channel', 'spatial', 'both', 'none'
        self.fusion_mode = kwargs.get('fusion_mode', 'concat')  # Default: 'concat', Options: 'add', 'gate', 'weighted_sum', 'add_linear', 'multiply'
        self.channel_memory_size = kwargs.get('channel_memory_size', 256)
        self.spatial_memory_size = kwargs.get('spatial_memory_size', 256)
        
        # Initialize memory modules based on mode
        self.use_channel_memory = self.memory_mode in ['channel', 'both']
        self.use_spatial_memory = self.memory_mode in ['spatial', 'both']
        
        self.channel_memory_module = None
        self.spatial_memory_module = None
        
        if self.use_channel_memory:
            # Channel memory module - xử lý feature patterns
            self.channel_memory_module = ChannelMemoryModule(
                mem_dim=self.channel_memory_size,
                feature_dim=hidden_dim,
                **kwargs
            )
            print('use channel mem')
        else:
            print('no channel mem')
        
        if self.use_spatial_memory:
            # Spatial memory module - xử lý spatial patterns  
            self.spatial_memory_module = SpatialMemoryModule(
                mem_dim=self.spatial_memory_size,
                height=feature_size[0],
                width=feature_size[1],
                **kwargs
            )
            print('use spatial mem')
        else:
            print('no spatial mem')
       
        self.transformer = Transformer( hidden_dim, feature_size, neighbor_mask, nhead=kwargs.get('nhead', 8), num_encoder_layers=kwargs.get('num_encoder_layers', 4), num_decoder_layers=kwargs.get('num_decoder_layers', 4), dim_feedforward=kwargs.get('dim_feedforward', 1024), dropout=kwargs.get('dropout', 0.1), activation=kwargs.get('activation', 'relu'), normalize_before=kwargs.get('normalize_before', False), )
        
        # Feature fusion layer - adapt based on memory mode and fusion method
        fusion_input_dim = hidden_dim
        if self.use_channel_memory and self.use_spatial_memory:
            if self.fusion_mode == 'concat':
                print("fusion mode: concat")
                fusion_input_dim = hidden_dim * 2
                self.fusion_layer = nn.Linear(fusion_input_dim, hidden_dim)
            elif self.fusion_mode == 'add' or self.fusion_mode == 'multiply':
                print("fusion mode: add/multiply")
                # No parameters needed for these operations
                self.fusion_layer = nn.Identity()
            elif self.fusion_mode == 'add_linear':
                print("fusion mode: add + linear")
                self.fusion_layer = nn.Linear(hidden_dim, hidden_dim)

            elif self.fusion_mode == 'weighted_sum':
                print("fusion mode: weighted sum (learnable alpha)")
                # Learnable fusion weight α
                self.alpha = nn.Parameter(torch.tensor(0.5))
                self.fusion_layer = nn.Identity()
            elif self.fusion_mode == 'gate':
                print("fusion mode: gate")
                # Gated fusion mechanism
                self.gate_layer = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Sigmoid()
                )
                self.fusion_layer = nn.Identity()
            else:
                print("fusion mode: unknown")
                raise ValueError(f"Unsupported fusion mode: {self.fusion_mode}")
        elif self.use_channel_memory or self.use_spatial_memory:
            # Single memory type - no fusion needed
            self.fusion_layer = nn.Identity()
        else:
            # No memory - no fusion needed
            self.fusion_layer = nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, inplanes[0])
        
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
    def add_random_mask(self, feature_tokens, mask_ratio, prob):
        num_tokens, batch_size, dim_channel = feature_tokens.shape
        
        for b in range(batch_size):
            if random.uniform(0, 1) <= prob:  # ← Check CHO TỪNG sample
                num_masked = int(num_tokens * mask_ratio)
                mask_indices = torch.randperm(num_tokens)[:num_masked]
                
                # Set masked tokens to zero
                feature_tokens[mask_indices, b, :] = 0.0
                
        return feature_tokens

    def forward(self, input):
        feature_align = input["feature_align"]  # B x C X H x W
        feature_tokens = rearrange(
            feature_align, "b c h w -> (h w) b c"
        )  # (H x W) x B x C
        
        # Add jitter during training if enabled
        if self.training and self.feature_jitter:
            feature_tokens = self.add_jitter(
                feature_tokens, self.feature_jitter.scale, self.feature_jitter.prob
            )
        # Add random masking during training if enabled
        # if self.training and self.feature_masking:
        #     # print('apply feature masking', self.feature_masking.get('ratio', 0.15), self.feature_masking.get('prob', 0.5))
        #     feature_tokens = self.add_random_mask(
        #         feature_tokens, 
        #         self.feature_masking.get('ratio', 0.15),  # Default 15% masking
        #         self.feature_masking.get('prob', 0.5)     # Default 50% probability
        #     )

        # Project input features
        feature_tokens = self.input_proj(feature_tokens)  # (H x W) x B x C
        feature_tokens = F.layer_norm(feature_tokens, feature_tokens.shape[-1:])
        # x_min, x_max = feature_tokens.min(), feature_tokens.max()
        # feature_tokens = (feature_tokens - x_min) / (x_max - x_min + 1e-6)
        # x_min, x_max = feature_tokens.min(), feature_tokens.max()
        # feature_tokens = 2 * (feature_tokens - x_min) / (x_max - x_min + 1e-6) - 1
        # mean = feature_tokens.mean()
        # std = feature_tokens.std()
        # feature_tokens = (feature_tokens - mean) / (std + 1e-6)
        # Get positional embeddings
        pos_embed = self.pos_embed(feature_tokens)  # (H x W) x C
        
        # Encode features using transformer encoder
        # encoded_tokens = self.encoder(
        #     feature_tokens, pos=pos_embed
        # )  # (H x W) x B x C
        encoded_tokens, mask_enc, mask_dec1, mask_dec2, pos_embed_batch = self.transformer( feature_tokens, pos_embed )
        # print('encoded_tokens: ', encoded_tokens.shape)
        # Memory retrieval based on memory mode
        memory_features_list = []
        channel_result = None
        spatial_result = None
        
        if self.use_channel_memory:
            channel_result = self.channel_memory_module(encoded_tokens)
            channel_retrieved = channel_result['output']  # (H x W) x B x C
            memory_features_list.append(channel_retrieved)
        
        if self.use_spatial_memory:
            spatial_result = self.spatial_memory_module(encoded_tokens)
            spatial_retrieved = spatial_result['output']  # C x B x H x W
            spatial_retrieved = torch.permute(spatial_retrieved, (2, 1, 0))  # (H x W) x B x C
            memory_features_list.append(spatial_retrieved)
        
        # Fuse memory features based on available memories
        if len(memory_features_list) == 0:
            # No memory - use encoded tokens directly
            memory_features = encoded_tokens
        elif len(memory_features_list) == 1:
            # Single memory type
            memory_features = memory_features_list[0]
        else:
            # Multiple memory types - fuse based on fusion_mode
            channel_features = memory_features_list[0]
            spatial_features = memory_features_list[1]
            
            if self.fusion_mode == 'concat':
                # Concatenation-based fusion (original method)
                combined_features = torch.cat([channel_features, spatial_features], dim=-1)  # (H x W) x B x (2*C)
                memory_features = self.fusion_layer(combined_features)  # (H x W) x B x C
            
            elif self.fusion_mode == 'add':
                # Addition-based fusion
                memory_features = channel_features + spatial_features
            
            elif self.fusion_mode == 'multiply':
                # Multiplication-based fusion
                memory_features = channel_features * spatial_features
            
            elif self.fusion_mode == 'add_linear':
                # Add rồi qua Linear để học mapping mới
                added_features = channel_features + spatial_features
                memory_features = self.fusion_layer(added_features)

            elif self.fusion_mode == 'weighted_sum':
                # Weighted sum với hệ số alpha có thể học
                memory_features = self.alpha * channel_features + (1 - self.alpha) * spatial_features

            elif self.fusion_mode == 'gate':
                combined = torch.cat([channel_features, spatial_features], dim=-1)  # [B, N, 2C]
                gate = self.gate_layer(combined)  # [B, N, C]
                memory_features = gate * channel_features + (1 - gate) * spatial_features

        # Add jitter AFTER memory retrieval/fusion if enabled
        # if self.training and self.feature_jitter:
        #     memory_features = self.add_jitter(
        #         memory_features, self.feature_jitter.scale, self.feature_jitter.prob
        #     )
        # Decode features
        # decoded_tokens = self.decoder(
        #     memory_features, 
        #     encoded_tokens, 
        #     pos=pos_embed
        # )  # (H x W) x B x C
        decoded_tokens = self.transformer.decoder(memory_features, memory_features, tgt_mask=mask_dec1, memory_mask=mask_dec2, pos=pos_embed_batch)
        
        # Project back to original dimension
        feature_rec_tokens = self.output_proj(decoded_tokens)  # (H x W) x B x C
        feature_rec_tokens = torch.sigmoid(feature_rec_tokens)
        # feature_rec_tokens = F.layer_norm(feature_rec_tokens, feature_rec_tokens.shape[-1:])
        # x_min, x_max = feature_rec_tokens.min(), feature_rec_tokens.max()
        # feature_rec_tokens = 2 * (feature_rec_tokens - x_min) / (x_max - x_min + 1e-6) - 1
        # mean = feature_rec_tokens.mean()
        # std = feature_rec_tokens.std()
        # feature_rec_tokens = (feature_rec_tokens - mean) / (std + 1e-6)
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
        feature_align = torch.sigmoid(feature_align) 
        # feature_align = F.layer_norm(feature_align, feature_align.shape[1:])
        # x_min, x_max = feature_align.min(), feature_align.max()
        # feature_align = 2 * (feature_align - x_min) / (x_max - x_min + 1e-6) - 1
        # mean = feature_align.mean()
        # std = feature_align.std()
        # feature_align = (feature_align - mean) / (std + 1e-6)
        pred = torch.sqrt(
            torch.sum((feature_rec - feature_align) ** 2, dim=1, keepdim=True)
        )  # B x 1 x H x W
        
        pred = self.upsample(pred)  # B x 1 x H x W
        
        # Prepare output dictionary based on available memories
        output_dict = {
            "feature_rec": feature_rec,
            "feature_align": feature_align,
            "pred": pred,
        }
        
        # Add memory-specific outputs if available
        if channel_result is not None:
            output_dict.update({
                "channel_attention": channel_result['att_weight'],
                "channel_scores": channel_result['attention_scores'],
            })
        
        if spatial_result is not None:
            output_dict.update({
                "spatial_attention": spatial_result['att_weight'],
                "spatial_ssim": spatial_result['ssim_similarity'],
            })
        
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
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    elif pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed