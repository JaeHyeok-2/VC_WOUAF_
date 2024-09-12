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

    def add_affine_attn(vaed):
        for layer in vaed.children():
            if layer.__class__.__name__ == "AttnBlock":
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
    def change_forward(vaed, target, new_forward):
        for name, layer in vaed.named_children():
            # 클래스 타입인 경우: 클래스 이름을 비교
            if isinstance(target, type):
                if layer.__class__.__name__ == target.__name__:
                    bound_method = new_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)

            # 문자열인 경우: 레이어의 이름을 비교
            elif isinstance(target, str):
                if name == target:
                    bound_method = new_forward.__get__(layer, layer.__class__)
                    setattr(layer, 'forward', bound_method)

            # 자식 레이어에 대해 재귀적으로 처리
            else:
                change_forward(layer, target, new_forward)

    def new_forward_MB(self, hidden_states, encoded_fingerprint, temb=None):
        hidden_states = self.resnets[0]((hidden_states, encoded_fingerprint), temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = attn((hidden_states, encoded_fingerprint))
            hidden_states = resnet((hidden_states, encoded_fingerprint), temb)

        return hidden_states

    def new_forward_UDB(self, hidden_states, encoded_fingerprint):
        for resnet in self.resnets:
            hidden_states = resnet((hidden_states, encoded_fingerprint), temb=None)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

    def new_forward_RB(self, input_tensor, temb):
        print(input_tensor.shape, temb.shape)
        input_tensor, encoded_fingerprint = input_tensor
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        phis = self.affine1(encoded_fingerprint)
        batch_size = phis.shape[0]
        weight = phis.view(batch_size, 1, -1, 1, 1) * self.conv1.weight.unsqueeze(0)
        hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv1.bias.view(1, -1, 1, 1)

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
        hidden_states = F.conv2d(hidden_states.contiguous().view(1, -1, hidden_states.shape[-2], hidden_states.shape[-1]), weight.view(-1, weight.shape[-3], weight.shape[-2], weight.shape[-1]), padding=1, groups=batch_size).view(batch_size, weight.shape[1], hidden_states.shape[-2], hidden_states.shape[-1]) + self.conv2.bias.view(1, -1, 1, 1)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

    def new_forward_AB(self, hidden_states):
        hidden_states, encoded_fingerprint = hidden_states
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        phis_q = self.affine_q(encoded_fingerprint)
        query_proj = torch.bmm(hidden_states, phis_q.unsqueeze(-1) * self.query.weight.t().unsqueeze(0)) + self.query.bias

        phis_k = self.affine_k(encoded_fingerprint)
        key_proj = torch.bmm(hidden_states, phis_k.unsqueeze(-1) * self.key.weight.t().unsqueeze(0)) + self.key.bias

        phis_v = self.affine_v(encoded_fingerprint)
        value_proj = torch.bmm(hidden_states, phis_v.unsqueeze(-1) * self.value.weight.t().unsqueeze(0)) + self.value.bias

        scale = 1 / math.sqrt(self.channels / self.num_heads)

        query_proj = self.reshape_heads_to_batch_dim(query_proj)
        key_proj = self.reshape_heads_to_batch_dim(key_proj)
        value_proj = self.reshape_heads_to_batch_dim(value_proj)

        if self._use_memory_efficient_attention_xformers:
            # Memory efficient attention
            hidden_states = xformers.ops.memory_efficient_attention(
                query_proj, key_proj, value_proj, attn_bias=None, op=self._attention_op
            )
            hidden_states = hidden_states.to(query_proj.dtype)
        else:
            attention_scores = torch.baddbmm(
                torch.empty(
                    query_proj.shape[0],
                    query_proj.shape[1],
                    key_proj.shape[1],
                    dtype=query_proj.dtype,
                    device=query_proj.device,
                ),
                query_proj,
                key_proj.transpose(-1, -2),
                beta=0,
                alpha=scale,
            )
            attention_probs = torch.softmax(attention_scores.float(), dim=-1).type(attention_scores.dtype)
            hidden_states = torch.bmm(attention_probs, value_proj)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    # Reference: https://github.com/huggingface/diffusers
    def new_forward_vaed(self, z, encoded_fingerprint):
        sample = z
        print("1123:", sample.shape)
        sample = self.conv_in(sample)
        print("1124:", sample.shape)
        # middle
        # sample = self.mid_block(sample, encoded_fingerprint)
        # 레이어들 개별적으로 호출
        sample = self.mid.block_1(sample, encoded_fingerprint)  # 첫 번째 Resnet 블록 호출
        sample = self.mid.attn_1(sample)  # Attention 레이어 호출
        sample = self.mid.block_2(sample, encoded_fingerprint)  # 두 번째 Resnet 블록 호출


        # up
        # for up_block in self.up_blocks:
        #     sample = up_block(sample, enconded_fingerprint)

        # # post-process
        # sample = self.conv_norm_out(sample)
        # sample = self.conv_act(sample)
        # sample = self.conv_out(sample)

        # return sample
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                sample = self.up[i_level].block[i_block](sample, encoded_fingerprint)
                
                # 어텐션이 있을 경우에만 적용
                if len(self.up[i_level].attn) > 0:
                    sample = self.up[i_level].attn[i_block](sample)
                
            # 레벨이 0이 아니면 업샘플링 적용
            if i_level != 0:
                sample = self.up[i_level].upsample(sample)

        # end 부분 적용
        if self.give_pre_end:
            return sample

        sample = self.conv_norm_out(sample)  # normalization 처리
        sample = self.conv_act(sample)       # activation 처리
        sample = self.conv_out(sample)       # convolution 처리

        # tanh 함수가 필요하면 적용
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
    change_forward(vae.decoder, "mid", new_forward_MB)
    change_forward(vae.decoder, "up", new_forward_UDB)
    change_forward(vae.decoder, ResnetBlock, new_forward_RB)
    change_forward(vae.decoder, AttnBlock, new_forward_AB)
    setattr(vae.decoder, 'forward', new_forward_vaed.__get__(vae.decoder, vae.decoder.__class__))
    setattr(vae, '_decode', new__decode.__get__(vae, vae.__class__))
    setattr(vae, 'decode', new_decode.__get__(vae, vae.__class__))

    return vae