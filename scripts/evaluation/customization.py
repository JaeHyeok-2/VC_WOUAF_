import torch
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers.utils import BaseOutput
from typing import Any, Dict, List, Optional, Tuple, Union
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, UpDecoderBlock2D, CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2DCrossAttn, UpBlock2D, CrossAttnUpBlock2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.attention import AttentionBlock
from diffusers.models.cross_attention import CrossAttention
from attribution import FullyConnectedLayer
import math
from lvdm.modules.networks.ae_modules import *


def customize_vae_decoder(vae, phi_dimension, lr_multiplier):
    def add_affine_conv(vaed):
        for layer in vaed.children():
            if layer.__class__.__name__ == "ResnetBlock":
                layer.affine1 = FullyConnectedLayer(phi_dimension, layer.conv1.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine2 = FullyConnectedLayer(phi_dimension, layer.conv2.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_conv(layer)

    # def add_affine_attn(vaed):
    #     for layer in vaed.children():
    #         if layer.__class__.__name__ == "AttnBlock":
    #             layer.affine_q = FullyConnectedLayer(phi_dimension, layer.q.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
    #             layer.affine_k = FullyConnectedLayer(phi_dimension, layer.k.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
    #             layer.affine_v = FullyConnectedLayer(phi_dimension, layer.v.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
    #         else:
    #             add_affine_attn(layer)
    def add_affine_attn(vaed):
        for layer in vaed.children():
            # 만약 layer가 ModuleList라면 그 안에 있는 모듈도 순회
            if isinstance(layer, nn.ModuleList):
                for sub_layer in layer:
                    add_affine_attn(sub_layer)
            elif layer.__class__.__name__ == "AttnBlock":
                layer.affine_q = FullyConnectedLayer(phi_dimension, layer.q.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_k = FullyConnectedLayer(phi_dimension, layer.k.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
                layer.affine_v = FullyConnectedLayer(phi_dimension, layer.v.weight.shape[1], lr_multiplier=lr_multiplier, bias_init=1)
            else:
                add_affine_attn(layer)

    # change_forward(vae.decoder, UNetMidBlock2D, new_forward_MB)
    # change_forward(vae.decoder, UpDecoderBlock2D, new_forward_UDB)

    # def change_forward(vaed, layer_type, new_forward):
    #     for layer in vaed.children():
    #         if type(layer) == layer_type:
    #             bound_method = new_forward.__get__(layer, layer.__class__)
    #             setattr(layer, 'forward', bound_method)
    #         else:
    #             change_forward(layer, layer_type, new_forward)
    
    # change_forward(vae.decoder, "mid", new_forward_MB)
    # change_forward(vae.decoder, "up", new_forward_UDB)
    # change_forward(vae.decoder, ResnetBlock, new_forward_RB)
    # change_forward(vae.decoder, AttnBlock, new_forward_AB)
    def change_forward(vaed, layer_type, new_forward):
        for layer in vaed.children():
            
            if type(layer) == layer_type:
                # print('LayerType Same!!',type(layer), layer_type)
                bound_method = new_forward.__get__(layer, layer.__class__)
                
                setattr(layer, 'forward', bound_method)
            else:
                change_forward(layer, layer_type, new_forward)
    
    def change_forward_mid_up(vaed, layer_type, new_forward):
        for name, layer in vaed.named_children():
            # 레이어의 이름이 layer_type과 같은지 확인
            if name == layer_type:
                # forward 메서드 교체
                bound_method = new_forward.__get__(layer, layer.__class__)
                setattr(layer, 'forward', bound_method)
            else:
                # 재귀적으로 내부 레이어 확인
                change_forward_mid_up(layer, layer_type, new_forward)

    # def new_forward_MB(self, hidden_states, encoded_fingerprint, temb=None):
    #     hidden_states = self.resnets[0]((hidden_states, encoded_fingerprint), temb)
    #     for attn, resnet in zip(self.attentions, self.resnets[1:]):
    #         if attn is not None:
    #             hidden_states = attn((hidden_states, encoded_fingerprint))
    #         hidden_states = resnet((hidden_states, encoded_fingerprint), temb)

    #     return hidden_states
    
    def new_forward_MB(self, hidden_states, encoded_fingerprint, temb=None):
        hidden_states = self.resnet[0]((hidden_states, encoded_fingerprint), temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn((hidden_states, encoded_fingerprint))
            hidden_states = resnet((hidden_states, encoded_fingerprint), temb)

        return hidden_states    

    # def new_forward_UDB(self, hidden_states, encoded_fingerprint):
    #     for resnet in self.resnets:
    #         hidden_states = resnet((hidden_states, encoded_fingerprint), temb=None)

    #     if self.upsamplers is not None:
    #         for upsampler in self.upsamplers:
    #             hidden_states = upsampler(hidden_states)

    #     return hidden_states
    def new_forward_UDB(self, hidden_states, encoded_fingerprint):
        print('호출?')
        for resnet in self.block:
            hidden_states = resnet((hidden_states, encoded_fingerprint), temb=None)

        if hasattr(self, 'upsample') and self.upsample is not None:
            hidden_states = self.upsample(hidden_states)

        return hidden_states

    def new_forward_RB(self, input_tensor, temb):
        
        input_tensor, encoded_fingerprint = input_tensor
        hidden_states = input_tensor
        # print(input_tensor, temb)

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)


        phis = self.affine1(encoded_fingerprint)
        batch_size = phis.shape[0]
        weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
        # hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)
        
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        phis = self.affine2(encoded_fingerprint)
        batch_size = phis.shape[0]
        weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv2.weight.unsqueeze(0)
        # hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv2.bias.view(1, -1, 1, 1)
        hidden_states = self.conv2(hidden_states) 
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                input_tensor = self.conv_shortcut(input_tensor)

            else:
                input_tensor = self.nin_shortcut(input_tensor) 

        # output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        output_tensor = input_tensor + hidden_states
        return output_tensor

    def new_forward_AB(self, hidden_states):
        
        hidden_states, encoded_fingerprint = hidden_states
        # x = hidden_states

        residual = hidden_states

        batch, channel, height, width = hidden_states.shape
        print(batch ,channel, height, width)
        # norm
        hidden_states = self.norm(hidden_states)
        # query
        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2) # b, hw, c

        # Affine 변환을 적용한 phis_q, phis_k, phis_v 생성
        phis_q = self.affine_q(encoded_fingerprint)  # (batch, channels)
        phis_k = self.affine_k(encoded_fingerprint)  # (batch, channels)
        phis_v = self.affine_v(encoded_fingerprint)  # (batch, channels)



        # self.q.weight, self.k.weight, self.v.weight의 차원을 (channels, channels)로 변경
        q_weight_reshaped = self.q.weight.view(channel, -1)  # (channels, channels)
        k_weight_reshaped = self.k.weight.view(channel, -1)  # (channels, channels)
        v_weight_reshaped = self.v.weight.view(channel, -1)  # (channels, channels)

        # Query 연산: hidden_states와 phis_q, q_weight_reshaped의 배치 행렬 곱
        query_proj = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * q_weight_reshaped.t().unsqueeze(0)) + self.q.bias  # (batch, hw, channels)

        # Key 연산: hidden_states와 phis_k, k_weight_reshaped의 배치 행렬 곱
        key_proj = torch.bmm(hidden_states, phis_k.unsqueeze(-1) * k_weight_reshaped.t().unsqueeze(0)) + self.k.bias  # (batch, hw, channels)

        # Value 연산: hidden_states와 phis_v, v_weight_reshaped의 배치 행렬 곱
        value_proj = torch.bmm(hidden_states, phis_v.unsqueeze(-1) * v_weight_reshaped.t().unsqueeze(0)) + self.v.bias  # (batch, hw, channels

        # query => b h*w c 
        key_proj = key_proj.reshape(batch,channel, height* width)
        print(query_proj.shape, key_proj.shape)
        # # proj to q, k, v
        # phis_q = self.affine_q(encoded_fingerprint)
        # query_proj = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * self.q.weight.t().unsqueeze(0)) + self.q.bias

        # phis_k = self.affine_k(encoded_fingerprint)
        # key_proj = torch.bmm(hidden_states, phis_k.unsqueeze(-1) * self.k.weight.t().unsqueeze(0)) + self.k.bias

        
        # phis_v = self.affine_v(encoded_fingerprint)
        # value_proj = torch.bmm(hidden_states, phis_v.unsqueeze(-1) * self.v.weight.t().unsqueeze(0)) + self.v.bias

        w_ = torch.bmm(query_proj, key_proj) 
        w_ = w_ * (int(channel) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        value_proj = value_proj.reshape(batch, channel, height*width)
        print("Value, W_ Shape :", value_proj.shape, w_.shape)
        hidden_states = torch.bmm(value_proj, w_) 
        hidden_states = hidden_states.reshape(batch,channel,height, width) 
        
        hidden_states = self.proj_out(hidden_states)

        # scale = 1 / math.sqrt(self.channels / self.num_heads)


        # if self._use_memory_efficient_attention_xformers:
        #     # Memory efficient attention
        #     hidden_states = xformers.ops.memory_efficient_attention(
        #         query_proj, key_proj, value_proj, attn_bias=None, op=self._attention_op
        #     )
        #     hidden_states = hidden_states.to(query_proj.dtype)
        # else:
        #     attention_scores = torch.baddbmm(
        #         torch.empty(
        #             query_proj.shape[0],
        #             query_proj.shape[1],
        #             key_proj.shape[1],
        #             dtype=query_proj.dtype,
        #             device=query_proj.device,
        #         ),
        #         query_proj,
        #         key_proj.transpose(-1, -2),
        #         beta=0,
        #         alpha=scale,
        #     )
        #     attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
        #     hidden_states = torch.bmm(attention_probs, value_proj)

        # # reshape hidden_states
        # hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # # compute next hidden_states
        # hidden_states = self.proj_attn(hidden_states)

        # hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # # res connect and rescale
        # hidden_states = (hidden_states + residual) / self.rescale_output_factor

        return hidden_states + residual

    # Reference: https://github.com/huggingface/diffusers
    def new_forward_vaed(self, z, encoded_fingerprint):
        sample = z
        sample = self.conv_in(sample)

        # middle

        sample = self.mid.block_1((sample, encoded_fingerprint), None)
        sample = self.mid.attn_1((sample, encoded_fingerprint))
        sample = self.mid.block_2((sample, encoded_fingerprint), None)
        
        # up
        # for up_block in self.up_blocks:
        #     sample = up_block(sample, encoded_fingerprint)
                # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                sample = self.up[i_level].block[i_block]((sample, encoded_fingerprint), None)                
                if len(self.up[i_level].attn) > 0:
                    sample = self.up[i_level].attn[i_block]((sample, encoded_fingerprint))
                # print(f'decoder-up feat={h.shape}')
            if i_level != 0:
                sample = self.up[i_level].upsample(sample)
                # print(f'decoder-upsample feat={h.shape}')

        # post-process
        sample = self.norm_out(sample)
        sample = nonlinearity(sample)
        sample = self.conv_out(sample)
        
        if self.tanh_out:
            sample = torch.tanh(sample)

        return sample
    @dataclass
    class DecoderOutput(BaseOutput):
        """
        Output of decoding method.
        Args:
            sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Decoded output sample of the model. Output of the last layer of the model.
        """

        sample: torch.FloatTensor

    def new__decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, encoded_fingerprint)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def new_decode(self, z: torch.FloatTensor, encoded_fingerprint: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        # if self.use_slicing and z.shape[0] > 1:
        print("latent Z의 Shape :", z.shape)
        if z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice, encoded_fingerprint).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, encoded_fingerprint).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    add_affine_conv(vae.decoder)
    add_affine_attn(vae.decoder)
    # change_forward(vae.decoder, UNetMidBlock2D, new_forward_MB)
    # change_forward(vae.decoder, UpDecoderBlock2D, new_forward_UDB)
    change_forward(vae.decoder, ResnetBlock, new_forward_RB)
    change_forward(vae.decoder, AttnBlock, new_forward_AB)
    change_forward_mid_up(vae.decoder, "mid", new_forward_MB)
    change_forward_mid_up(vae.decoder, "up", new_forward_UDB)

    setattr(vae.decoder, 'forward', new_forward_vaed.__get__(vae.decoder, vae.decoder.__class__))
    setattr(vae, '_decode', new__decode.__get__(vae, vae.__class__))
    setattr(vae, 'decode', new_decode.__get__(vae, vae.__class__))

    return vae