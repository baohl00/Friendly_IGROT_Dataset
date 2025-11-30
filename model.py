import torch 
import torch.nn as nn
from clip import clip
from transformer import *
import torch.nn.functional as F
from layers import *
from transformers.models.t5.modeling_t5 import T5Block, T5Stack, T5LayerCrossAttention 
from transformers.models.t5 import T5Config
from BLIP.models.blip_retrieval import blip_retrieval
#from Qformer import BertConfig, BertLMHeadModel
import open_clip

MODEL_PATH = "/home/hle/spinning-storage/hle/ckpt" 

class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, combine_feat, fuse_feat):
        # Reshape to [B, 1, D] for attention
        q = combine_feat.unsqueeze(1)  # query: combine_features
        k = v = fuse_feat.unsqueeze(1)  # keys/values: fuse_features

        output, _ = self.cross_attn(q, k, v)  # [B, 1, D]
        fused = self.norm(output + q)  # Residual connection
        fused = self.proj(fused.squeeze(1))  # Optional projection
        return F.normalize(fused, dim=-1)  # [B, D]

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()  # outputs weights in [0, 1]
        )

    def forward(self, combine_feat, fuse_feat):
        # [B, D] → [B, 2D]
        concat_feat = torch.cat([combine_feat, fuse_feat], dim=-1)
        
        # compute gate values
        gate = self.gate_layer(concat_feat)  # [B, D]
        
        # blend the features
        output = gate * combine_feat + (1 - gate) * fuse_feat
        return F.normalize(output, dim=-1)

class TransAgg(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg.device  
        self.main_type = cfg.main_type
        self.target_type = cfg.target_type 
        self.model_name = cfg.model
        self.inference = cfg.vision_projector 
        if self.model_name == 'blip':
            self.model = blip_retrieval(pretrained=f"{MODEL_PATH}/model_base_retrieval_coco.pth")
            self.feature_dim = 256
        elif self.model_name == 'blip_large':
            self.model = blip_retrieval(pretrained=f"{MODEL_PATH}/model_large_retrieval_coco.pth")
            self.feature_dim = 256
        elif self.model_name == "blip_flickr":
            self.model = blip_retrieval(pretrained=f"{MODEL_PATH}/model_base_retrieval_flickr.pth")
            self.feature_dim = 256
        elif self.model_name == 'clip_base':
            self.model, self.preprocess = clip.load(f"{MODEL_PATH}/ViT-B-32.pt", device=cfg.device, jit=False)
            self.feature_dim = self.model.visual.output_dim 
        elif self.model_name == 'clip_large':
            self.model, self.preprocess = clip.load(f"{MODEL_PATH}/ViT-L-14.pt", device=cfg.device, jit=False)
            self.feature_dim = self.model.visual.output_dim 
        # elif self.model_name == 'remote_clip':
        #     self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', device=cfg.device)
            
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=8, dropout=cfg.dropout, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.logit_scale = 100
        self.dropout = nn.Dropout(cfg.dropout)
        self.combiner_layer = nn.Linear(self.feature_dim + self.feature_dim, (self.feature_dim + self.feature_dim) * 4)
        self.weighted_layer = nn.Linear(self.feature_dim, 3)
        self.output_layer = nn.Linear((self.feature_dim + self.feature_dim) * 4, self.feature_dim)
        self.sep_token = nn.Parameter(torch.randn(1, 1, self.feature_dim))
        self.union_linear = nn.Linear(self.feature_dim, self.feature_dim * 3)
        self.linear = nn.Linear(self.feature_dim * 3, self.feature_dim * 3)
        self.attention = nn.MultiheadAttention(
                embed_dim = self.feature_dim, 
                num_heads = 8 if 'large' in self.model_name else 4, 
                batch_first = True,
                kdim = self.feature_dim,
                vdim = self.feature_dim,
                dropout = 0.0,
                bias = True
        )
        # self.last_linear = nn.Linear(self.feature_dim * 3, self.feature_dim)

        self.dynamic_scalar = nn.Sequential(
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, 2),
            nn.Softmax(dim=-1)
        )
        # self.gated_fusion = GatedFusion(self.feature_dim) 

        self.cross_attention = CrossModalFusion(self.feature_dim, num_heads=8 if 'large' in self.model_name else 4)

        self.union_weighted_layer = nn.Sequential(
                nn.Linear(self.feature_dim * 2, self.feature_dim * 4), 
                nn.Dropout(cfg.dropout/2), 
                nn.Sigmoid(),
                nn.Linear(self.feature_dim * 4, self.feature_dim), 
                nn.Dropout(cfg.dropout/2), 
                nn.Sigmoid(), 
                nn.Linear(self.feature_dim, 1))
        
        # T5 layers for feature fusion
        if cfg.model in ["clip_base", "blip", "blip_flickr"]:
            conf_t5 = T5Config()
            conf_t5.num_layers = 4
            conf_t5.num_decoder_layers = 4
            conf_t5.num_heads = 8
            conf_t5.d_model = self.feature_dim 
            conf_t5.d_kv = 64
            conf_t5.feed_forward_proj = "relu"
            self.t5_layers = T5Stack(conf_t5)
        elif cfg.model in ["clip_large", "blip_large"]:
            conf_t5_vit_large = T5Config()
            conf_t5_vit_large.num_layers = 4
            conf_t5_vit_large.num_decoder_layers = 4
            conf_t5_vit_large.num_heads = 12
            conf_t5_vit_large.d_model = self.feature_dim
            conf_t5_vit_large.d_kv = 64
            conf_t5_vit_large.feed_forward_proj = "relu"
            self.t5_layers = T5Stack(conf_t5_vit_large)
        else:
            raise NotImplementedError("Only ViT-B/32, ViT-L/14 and BLIP are supported.")
        
        self.reduced_dim = self.feature_dim // 2

    def forward(self, texts, reference_images, target_images, reference_captions = None, target_captions = None):
        img_text_rep = self.final_features(reference_images, texts) 
        
        null_texts = [""] * len(target_images) 
        if self.model_name.startswith('blip'):
            tokenized_null_texts = self.model.tokenizer(null_texts, padding='max_length', truncation=True, max_length=35,
                                                              return_tensors='pt').to(self.device)
        elif self.model_name.startswith('clip'):
            tokenized_null_texts = clip.tokenize(null_texts, truncate=True).to(self.device)
        
        text_target_features, _ = self.model.encode_text(tokenized_null_texts)
        # target_features, _ = self.model.encode_image(target_images)
        # target_features = F.normalize(target_features, dim=-1)
        target_features = self.target_features(target_images)
        aux_ref_features, _ = self.model.encode_image(reference_images)
        if self.target_type == "sum":
            # target_features += F.normalize(text_target_features, dim = -1)
            aux_ref_features = self.model.encoded_image(reference_images, return_local=True)[0] + F.normalize(self.model.encode_text(tokenized_null_texts, return_local=True)[0], dim = -1)
            # aux_tar_features = self.union_features(target_images, texts)
        elif self.target_type == "union":
            # target_features = self.union_features(target_images, null_texts)
            aux_ref_features = self.union_features(reference_images, null_texts)
            # aux_tar_features = self.union_features(target_images, texts)
        elif self.target_type == "fuse":
            # target_features = self.fuse_features(target_images, null_texts)
            aux_ref_features = self.fuse_features(reference_images, null_texts)
            # aux_tar_features = self.combine_features(target_images, texts)
        
        return img_text_rep, target_features, aux_ref_features
    
    def final_features(self, reference_images, texts):
        if self.main_type == "union":
            return self.union_features(reference_images, texts)
        elif self.main_type == "fuse":
            return self.fuse_features(reference_images, texts)
        elif self.main_type == "combine":
            return self.combine_features(reference_images, texts)
        else:
            raise NotImplementedError("Only union, fuse and combine are supported.")
    
    def target_features(self, target_images):
        null_texts = [""] * len(target_images) 
        if self.model_name.startswith('blip'):
            tokenized_null_texts = self.model.tokenizer(null_texts, padding='max_length', truncation=True, max_length=35,
                                                              return_tensors='pt').to(self.device)
        elif self.model_name.startswith('clip'):
            tokenized_null_texts = clip.tokenize(null_texts, truncate=True).to(self.device)
        
        text_target_features, _ = self.model.encode_text(tokenized_null_texts)
        target_features, _ = self.model.encode_image(target_images)
        target_features = F.normalize(target_features, dim=-1)
        if self.target_type == "sum":
            target_features += F.normalize(text_target_features, dim=-1)
            # aux_tar_features = self.union_features(target_images, texts)
        elif self.target_type == "union":
            target_features = self.union_features(target_images, null_texts)
        elif self.target_type == "fuse":
            target_features = self.fuse_features(target_images, null_texts)
        
        return target_features 

    def union_features(self, reference_images, texts):
        img_embeds, _ = self.model.encode_image(reference_images, return_local = True) # [B, D]
        if self.model_name.startswith('blip'):
            tokenized_texts = self.model.tokenizer(texts, padding='max_length', truncation=True, max_length=35,
                    return_tensors='pt').to(self.device)
        elif self.model_name.startswith('clip'):
            tokenized_texts = clip.tokenize(texts, truncate = True).to(self.device)
        txt_embeds, _ = self.model.encode_text(tokenized_texts) # [B, D] 
        concat = torch.cat((img_embeds.unsqueeze(1), txt_embeds.unsqueeze(1)), dim=1) # [B, 2, D]
        transformer_embed = self.t5_layers(
                inputs_embeds = concat, 
                attention_mask = None,
                use_cache = False,
                return_dict = True
                )
        w = transformer_embed.last_hidden_state # [B, 2, D] 
        i_w, t_w = w[:, 0] , w[:, 1]
        union_feats = torch.cat((i_w, t_w), dim = -1)
        union_weighted = self.union_weighted_layer(union_feats) # [B, 1]
        
        output_rep = union_weighted[:, 0:1] * img_embeds + (1 - union_weighted[:, 0:1]) * txt_embeds

        output_rep = F.normalize(output_rep, dim = -1) # [B, D] 

        return output_rep


    def fuse_features(self, images, texts, reference_captions = None):
        def topk_variance_mask(embeddings, top_k_ratio=0.2):
            """
            Selects top-k dimensions based on variance across batch.
            
            Args:
                embeddings: Tensor of shape [B, D]
                top_k_ratio: float, fraction of dimensions to keep
            
            Returns:
                mask: Tensor of shape [B, D] with 1s in top-k variance dims
            """
            B, D = embeddings.shape
            # Compute variance across batch
            variances = torch.var(embeddings, dim=0)  # [D]
            
            # Get top-k indices
            top_k = int(top_k_ratio * D)
            _, top_k_indices = torch.topk(variances, top_k)

            # Create binary mask
            mask = torch.zeros(D, device=embeddings.device)
            mask[top_k_indices] = 1.0
            mask = mask.unsqueeze(0).expand(B, -1)  # [B, D]
        
            return mask
        
        union_feat = self.union_features(images, texts) 

        fuse = union_feat 
        alpha = nn.Sigmoid()(fuse)  
        mask = topk_variance_mask(fuse)  # [B, 3D]
        alpha = alpha * mask  # Apply the mask to alpha
        alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
        fuse = alpha * fuse + fuse  # [B, 3D]
        
        query_rep = fuse
        
        query_rep = F.normalize(query_rep, dim=-1)  # [B, D]
        
        return query_rep

    def combine_features(self, reference_images, texts, reference_captions = None):
        reference_image_features, reference_total_image_features = self.model.encode_image(reference_images, return_local=True)
        batch_size = reference_image_features.size(0)
        reference_total_image_features = reference_total_image_features.float()
        
        if self.model_name.startswith('blip'):
            tokenized_texts = self.model.tokenizer(texts, padding='max_length', truncation=True, max_length=35, return_tensors='pt').to(self.device)
            mask = (tokenized_texts.attention_mask == 0)
        elif self.model_name.startswith('clip'):
            tokenized_texts = clip.tokenize(texts, truncate=True).to(self.device)
            mask = (tokenized_texts == 0)
        text_features, total_text_features = self.model.encode_text(tokenized_texts)

        num_patches = reference_total_image_features.size(1)
        sep_token = self.sep_token.repeat(batch_size, 1, 1)

        combine_features = torch.cat((total_text_features, sep_token, reference_total_image_features), dim=1)

        image_mask = torch.zeros(batch_size, num_patches + 1).to(reference_image_features.device)
        mask = torch.cat((mask, image_mask), dim=1)
        
        img_text_rep = self.fusion(combine_features, src_key_padding_mask=mask) 
        
        if self.model_name.startswith('blip'): 
            multimodal_img_rep = img_text_rep[:, 36, :] 
            multimodal_text_rep = img_text_rep[:, 0, :]
        elif self.model_name.startswith('clip'):
            multimodal_img_rep = img_text_rep[:, 78, :]
            multimodal_text_rep = img_text_rep[torch.arange(batch_size), tokenized_texts.argmax(dim=-1), :]

        concate = torch.cat((multimodal_img_rep, multimodal_text_rep), dim=-1)
        f_U = self.output_layer(self.dropout(F.relu(self.combiner_layer(concate))))
        weighted = self.weighted_layer(f_U) # [B, 3]
        
        query_rep = weighted[:, 0:1] * text_features + weighted[:, 1:2] * f_U + weighted[:, 2:3] * reference_image_features
        
        query_rep = F.normalize(query_rep, dim=-1)

        return query_rep 
