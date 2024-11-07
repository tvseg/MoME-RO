# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock
from monai.networks.nets import ViT
from monai.utils import ensure_tuple_rep
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

from .modules import ContextUnetrUpBlock, UnetOutUpBlock, ExpertContextUnetrUpBlock
from .sam import TwoWayTransformer
from .decoder import TPN_DecoderLayer
from .llama2.llama_custom import LlamaForCausalLM
from .llama3.llama_custom import LlamaForCausalLM3
from .text_encoder import tokenize, TextContextEncoder
# from model.ConTEXTualSegmentation.models.LanguageCrossAttention import LangCrossAtt3D
# from transformers import T5Model, T5Tokenizer
from einops import rearrange
from functools import partial

from peft import (
        LoraConfig,
        get_peft_model,
        # prepare_model_for_int8_training,
    )
# from mamba_ssm import Mamba
from numpy.linalg import norm

cos_sim = lambda a,b: (a @ b.T) / (norm(a.detach().cpu())*norm(b.detach().cpu()))



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]

            return x

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                # bimamba_type="v2",
        )
    
    def forward(self, x):
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out
    
class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 48, 96, 192],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
              LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
              )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 4 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x
    
# network
class ContextUNETR(nn.Module):
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        feature_size: int = 24,
        norm_name: str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        context=False,
        args=None,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        Examples::
            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)
            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))
            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        # if feature_size % 12 != 0:
        #     raise ValueError("feature_size should be divisible by 12.")

        self.context = context
        self.normalize = normalize
        self.regularizer = args.regularizer
        self.noise = args.moe
        print("moe: ")
        print(self.noise)
        self.rag = args.rag
        self.stage = args.stage

        self.encoder1 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = in_channels,
            out_channels = feature_size,
            kernel_size =3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder2 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder3 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = feature_size,
            out_channels = 2 * feature_size ,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder4 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels= 2 * feature_size,
            out_channels = 4 * feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)

        self.encoder10 = UnetrBasicBlock(spatial_dims=spatial_dims,
            in_channels = 4 * feature_size,
            out_channels = 8 * feature_size,
            kernel_size = 3, stride=2, norm_name=norm_name, res_block=True)
        
        # decoder
        if args.target == 1:
            self.decoder4 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size * 8 ,
                out_channels = feature_size * 4,
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)

            self.decoder3 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size * 4,
                out_channels = feature_size * 2,
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)
            
            self.decoder2 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size * 2,
                out_channels = feature_size,
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

            self.decoder1 = ContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size,
                out_channels = feature_size, 
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)
        else:
            self.decoder4 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size * 8 ,
                out_channels = feature_size * 4,
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

            self.decoder3 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size * 4,
                out_channels = feature_size * 2,
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)
            
            self.decoder2 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size * 2,
                out_channels = feature_size,
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

            self.decoder1 = ExpertContextUnetrUpBlock(spatial_dims=spatial_dims,
                in_channels = feature_size,
                out_channels = feature_size, 
                kernel_size = 3, upsample_kernel_size=2, norm_name=norm_name)

        # out
        self.out = UnetOutUpBlock(spatial_dims=spatial_dims, 
            in_channels=feature_size, 
            out_channels=out_channels, 
            kernel_size=3, upsample_kernel_size=2, norm_name=norm_name, res_block=True)
        
        feature_size_list = [self.encoder1.layer.conv3.out_channels, self.encoder2.layer.conv3.out_channels, self.encoder3.layer.conv3.out_channels, self.encoder4.layer.conv3.out_channels, self.encoder10.layer.conv3.out_channels]

        # text encoder
        if self.context:

            self.lora = args.lora
            self.context_mode = args.context_mode
            
            if (args.textencoder == 'llama2') | (args.textencoder == 'llama3'):
                self.txt_embed_dim = 4096 #4096
            elif args.textencoder == 'llama2_13b':
                self.txt_embed_dim = 5120 #4096
            elif args.textencoder == 't5':
                self.txt_embed_dim = 1024
            else:
                self.txt_embed_dim = 512

            # align module (SAM)
            self.compare_mode = 0
            txt2vis, attntrans, matching = [], [], []
            for i in range(len(depths)+1):
                if i == 0:
                    matching.append(nn.Linear(self.txt_embed_dim, feature_size_list[i]))
                else:
                    matching.append(nn.Linear(feature_size_list[i], feature_size_list[i-1]))
                txt2vis.append(nn.Linear(self.txt_embed_dim, feature_size_list[i]))

                if args.compare_mode == 1:
                    attntrans.append(LangCrossAtt3D(emb_dim=feature_size_list[i]))
                    self.compare_mode = 1 #0
                else:
                    attntrans.append(TwoWayTransformer(depth=2,
                                                    embedding_dim=feature_size_list[i],
                                                    mlp_dim=feature_size*(2**i),
                                                    num_heads=8,
                                                    stage=args.stage,
                                                    expert=args.expert,
                                                    topk=args.topk,
                                                    ))
            self.txt2vis = nn.Sequential(*txt2vis)
            self.attn_transformer = nn.Sequential(*attntrans)
            
            # text encoder
            self.text_encoder = TextContextEncoder(embed_dim=self.txt_embed_dim, noise=args.noise, alpha=args.alpha)
            self.context_length = args.context_length
            self.token_embed_dim = self.text_encoder.text_projection.shape[-1]
            self.contexts = nn.Parameter(torch.randn(args.n_prompts, self.context_length, self.token_embed_dim))
            if self.rag:
                self.top_k = args.top_k
            self.max_length = 77

            for name, param in self.text_encoder.named_parameters():
                param.requires_grad_(False)
            
            # llama2
            if args.textencoder.find('llama') >= 0:
                
                self.text_encoder.llm = True
                rep_llama = '/Users/yo084/Documents/Projects/99_MoMCE-RO_vFinal/model/llama3/Meta-Llama-3-8B-Instruct'
                self.tokenizer = AutoTokenizer.from_pretrained(rep_llama)

                if args.flag_pc:
                    self.max_length = 64
                else:
                    self.max_length = 128 #64#128 #512 1024
                
                self.text_encoder.transformer  = LlamaForCausalLM3.from_pretrained(
                    rep_llama,
                    torch_dtype=torch.float16,
                    device_map="cpu", 
                ).model
                
                self.tokenizer._add_tokens(["<SEG>"], special_tokens=True)
                if args.flag_pc:
                    self.tokenizer._add_tokens(["<grade>"], special_tokens=True)
                    self.tokenizer._add_tokens(["<stage>"], special_tokens=True)
                    self.tokenizer._add_tokens(["<metastasis>"], special_tokens=True)
                    self.tokenizer._add_tokens(["<age>"], special_tokens=True)
                    self.tokenizer._add_tokens(["<psa>"], special_tokens=True)
                self.text_encoder.transformer.resize_token_embeddings(len(self.tokenizer) + (1 if args.flag_pc else 1)) #6
                self.text_encoder.token_embedding = self.text_encoder.transformer.embed_tokens
                
                for name, param in self.text_encoder.transformer.named_parameters():
                    param.requires_grad_(False)
                    
                if args.alpha:
                    self.alpha = nn.Parameter(torch.randn(1))
                    print(self.alpha)
                else:
                    self.alpha = None

                if args.compare_mode == 1:
                    self.text_encoder.all =  True

        if (args.stage == 3) & (args.max_epochs == 501):
            for name, param in self.named_parameters():
                if (name.find('attn_transformer')>=0) | (name.find('decoder')>=0) | (name.find('out.')>=0):  
                    print(name)
                else:
                    param.requires_grad_(False)

        self.ablation_none = (args.ablation == "None")
        if args.flag_pc:
            self.flag_pc = True
            if (args.context) & (args.textencoder == 'llama3'): # & (args.stage == 1):
                self.text_encoder.all = True 
        else:
            self.flag_pc = False

        
    def load_from(self, weights):
        pass

    def interactive_alignment(self, hidden_states_out, report_in, test_mode, x_in, status_train):
        
        tok_txt = []
        emb_txt = []
        emb_txt_t = []
            
        # prepare text tokens
        if self.text_encoder.llm:
            tok_txt = report_in
        else:
            for report in report_in:
                # if self.lora:
                #     tok_txt_ = self.tokenizer.encode(report)
                #     tok_txt_ = torch.tensor(tok_txt_).unsqueeze(0)
                #     emb_txt.append(self.text_encoder(tok_txt_.to(x_in.device)).unsqueeze(0)) # B, N, 
                if self.context_mode == 2:
                    report = report.replace(';','').replace(' ','').replace('<SEG>','')
                    tok_txt_= tokenize(report, self.max_length-self.context_length)[0, 1:1+len(report)]
                    m = nn.ConstantPad1d((0, max(0, 8-len(report))), 0)
                    tok_txt_ = m(torch.tensor(tok_txt_, dtype=torch.long))
                    tok_txt_ = self.text_encoder.token_embedding(tok_txt_.to(x_in.device))
                else:
                    tok_txt_= tokenize(report, self.max_length-self.context_length)
                tok_txt.append(tok_txt_)
            tok_txt = torch.stack(tok_txt, dim=0)

        if (self.context_mode == 2) | (self.compare_mode == 1) :
            emb_txt = tok_txt.to(x_in.device)
        else:
            emb_txt = self.text_encoder(tok_txt.to(x_in.device), self.contexts, train = True if status_train else False, alpha=self.alpha) 

        # projection
        report_l = []
        for i in self.txt2vis._modules.keys():
            report_l.append(self.txt2vis._modules[i](emb_txt))  

        # interactive alignment
        h_offset = 1 if self.flag_pc else 0 
        for j, text_vis in reversed(list(enumerate(zip(report_l[h_offset:], hidden_states_out[h_offset:])))):
            txt, vis = text_vis
            
            if len(report_in) != len(x_in):
                txt = torch.repeat_interleave(txt, vis.shape[0], dim=0)
            
            _, hidden_states_out[j+h_offset] = self.attn_transformer[j+h_offset](vis, None, txt, test_mode) #1 self.stage

        return hidden_states_out, emb_txt, emb_txt_t
        
    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x
    
    def forward(self, x_in, report_in=None, test_mode=None, target=None, mr=None):

        if mr is not None:
            x_in = torch.cat([x_in, mr], dim=1)

        hidden_states_out, score_logit = [], []

        # 3D UNet
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(enc0)
        enc2 = self.encoder3(enc1)
        enc3 = self.encoder4(enc2)
        dec4 = self.encoder10(enc3)
    
        hidden_states_out.append(enc0)
        hidden_states_out.append(enc1)
        hidden_states_out.append(enc2)
        hidden_states_out.append(enc3)
        hidden_states_out.append(dec4)

        # added context align by yujin @ 2303
        if self.context & (not self.ablation_none): 
            
            if self.noise == 0:
               test_mode = None
            elif self.noise == 1:
                test_mode = [0 for t in test_mode]
            # print(test_mode)

            hidden_states_out, emb_txt, ebd_txt_t = self.interactive_alignment(hidden_states_out, report_in, test_mode, x_in, status_train = target is not None)
   
        dec2 = self.decoder4(hidden_states_out[4], hidden_states_out[3])
        dec1 = self.decoder3(dec2, hidden_states_out[2])
        dec0 = self.decoder2(dec1, hidden_states_out[1])
        out = self.decoder1(dec0, hidden_states_out[0])

        logits = self.out(out)
        
        return logits
      

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=4, hidden=512):
        super().__init__()
        self.num_domains = num_domains

        layers = []
        layers += [nn.Linear(latent_dim, hidden)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden, hidden)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, hidden),
                                        nn.ReLU(),
                                        nn.Linear(hidden, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0)))
        s = out[idx, y.type(idx.dtype)]  # (batch)
        return s