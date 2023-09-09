"""
Created on Fri. Sept. 8 00:45:46 2023
@author: Dong Zhang
"""


import faulthandler
faulthandler.enable()


import random
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy
from icefall.utils import make_pad_mask


from uslm.data.input_strategies import PromptedFeatures
from uslm.modules.embedding import SinePositionalEmbedding, TokenEmbedding
from uslm.modules.transformer import (
    AdaptiveLayerNorm,
    LayerNorm,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from uslm.models.valle import VALLF, top_k_top_p_filtering, topk_sampling, Transpose

from .macros import NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS
from .visualizer import visualize


class USLM(VALLF):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        **kwargs,
    ):
        """
        Args:
          d_model:
            The number of expected features in the input (required).
          nhead:
            The number of heads in the multiheadattention models (required).
          num_layers:
            The number of sub-decoder-layers in the decoder (required).
        """
        super(USLM, self).__init__(
            d_model,
            nhead,
            num_layers,
            norm_first=norm_first,
            add_prenet=add_prenet,
            decoder_cls=TransformerEncoder,
            decoder_layer_cls=TransformerEncoderLayer,
            prefix_mode=prefix_mode,
            share_embedding=share_embedding,
            nar_scale_factor=nar_scale_factor,
            **kwargs,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: Union[torch.Tensor, PromptedFeatures],
        y_lens: Union[torch.Tensor, PromptedFeatures],
        reduction: str = "sum",
        train_stage: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Args:
        x:
            A 2-D tensor of shape (N, S).
        x_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        y:
            A 3-D tensor of shape (N, T, 8).
        y_lens:
            A 1-D tensor of shape (N,). It contains the number of tokens in `x`
            before padding.
        train_stage:
            0: AR & NAR modules, 1: AR modules, 2: NAR modules
        Returns:
        Return the predicted audio code matrix, cross-entropy loss and Top-10 accuracy.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape

        y_prompts_codes = None
        if isinstance(y, PromptedFeatures):
            y_prompts_codes, y = y.data
            prompts_len, y_lens = y_lens.data
            assert prompts_len.min() == prompts_len.max()
            assert self.prefix_mode == 4
            y_prompts_codes = y_prompts_codes.type(torch.int64)

        assert y.ndim == 3, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)
        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        text = x
        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        y, targets = self.pad_y_eos(
            codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
        )

        x_len = x_lens.max()

        metrics = {}
        total_loss = 0.0

        xy_padding_mask = torch.concat([x_mask, y_mask], dim=1)
        if self.ar_audio_prepend_bos:
            ar_xy_padding_mask = torch.concat(
                [x_mask, F.pad(y_mask, (1, 0), value=False)], dim=1
            )
        else:
            ar_xy_padding_mask = xy_padding_mask
        # AR Decoder
        if train_stage in [0, 1]:
            x = self.ar_text_embedding(text)
            x = self.ar_text_prenet(x)
            x = self.ar_text_position(x)

            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)

            x_attn_mask = F.pad(
                torch.zeros((x_len, x_len), dtype=torch.bool, device=x.device),
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool, device=x.device),
                    diagonal=1,
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat([x_attn_mask, y_attn_mask], dim=0)

            # merge key padding and attention masks
            bsz, src_len = x.shape[0], x_len + y_len
            _xy_padding_mask = (
                ar_xy_padding_mask.view(bsz, 1, 1, src_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(bsz * self.num_heads, 1, src_len)
            )
            xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)

            new_attn_mask = torch.zeros_like(xy_attn_mask, dtype=x.dtype)
            new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
            xy_attn_mask = new_attn_mask

            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            xy_pos = torch.concat([x, y_pos], dim=1)

            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
                # src_key_padding_mask=xy_padding_mask,
                # is_causal=True,
            )
            logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
            # loss
            total_loss = F.cross_entropy(logits, targets, reduction=reduction)

            metrics["ArTop10Accuracy"] = self.ar_accuracy_metric(
                logits.detach(), targets
            ).item() * y_lens.sum().type(torch.float32)

        if self.num_quantizers == 1:
            return ((x, codes), total_loss, metrics)

        # Non-AR Decoders
        if self.ar_audio_prepend_bos:
            y = y[:, 1:]
        if train_stage in [0, 2]:
            num_nar_layers = self.num_quantizers - 1

            #layer2: contain most timbre information
            nar_stage = 1

            x = self.nar_text_embedding(text)
            x = self.nar_text_prenet(x)
            x = self.nar_text_position(x)

            y_emb, prefix_len = self._prepare_prompts(
                y, y_lens, codes, nar_stage, y_prompts_codes
            )

            y_len = y_lens.max()
            targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            if self.prefix_mode in [2, 4]:
                xy_padding_mask = torch.concat(
                    [
                        x_mask,
                        F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),
                    ],
                    dim=1,
                )
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]

            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            xy_pos = torch.concat([x, y_pos], dim=1)
            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
                src_key_padding_mask=xy_padding_mask,
                # is_causal=False,
            )
            xy_dec = xy_dec[:, x_lens.max() + prefix_len :]
            if self.prefix_mode == 4:
                prefix_len = 0  # reset for Top10Accuracy metric
            logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(
                0, 2, 1
            )

            # loss
            total_length = (y_lens).sum().type(torch.float32)
            total_loss += (
                F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=NUM_AUDIO_TOKENS,
                    reduction=reduction,
                )
                * (total_length / (total_length - prefix_len * x.shape[0]))
            )
            metrics["NarL2Top10Accuracy"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )


            #layer3-8
            nar_stage = self.rng.choices(
                [_k for _k in range(2, self.num_quantizers)],
                weights=[1.0 / (num_nar_layers-1)] * (num_nar_layers-1),
                k=1,
            )[0]

            x = self.nar_text_embedding(text)
            x = self.nar_text_prenet(x)
            x = self.nar_text_position(x)

            y_emb, prefix_len = self._prepare_prompts(
                y, y_lens, codes, nar_stage, y_prompts_codes
            )

            y_len = y_lens.max()
            targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            if self.prefix_mode in [2, 4]:
                xy_padding_mask = torch.concat(
                    [
                        x_mask,
                        F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),
                    ],
                    dim=1,
                )
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]

            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)
            xy_pos = torch.concat([x, y_pos], dim=1)
            xy_dec, _ = self.nar_decoder(
                (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
                src_key_padding_mask=xy_padding_mask,
                # is_causal=False,
            )
            xy_dec = xy_dec[:, x_lens.max() + prefix_len :]
            if self.prefix_mode == 4:
                prefix_len = 0  # reset for Top10Accuracy metric
            logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(
                0, 2, 1
            )

            # loss
            total_length = (y_lens).sum().type(torch.float32)
            total_loss += (
                F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=NUM_AUDIO_TOKENS,
                    reduction=reduction,
                )
                * (total_length / (total_length - prefix_len * x.shape[0]))
            )
            metrics["NarL3-8Top10Accuracy"] = (
                self.nar_accuracy_metric(
                    F.pad(
                        logits.detach(),
                        (0, 0, 0, 1, 0, 0),
                        value=logits.min().cpu().item(),
                    ),
                    targets,
                ).item()
                * total_length
            )

        if train_stage == 0:
            total_loss = total_loss / 2.0

        return ((x, codes), total_loss, metrics)

    def inference(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        enroll_x_lens: torch.Tensor,
        top_k: int = -100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
        x:
            A 2-D tensor of shape (1, S).
        x_lens:
            A 1-D tensor of shape (1,). It contains the number of tokens in `x`
            before padding.
        y:
            A 3-D tensor of shape (1, T, 8).
        top_k: (`optional`) int
            The number of highest probability tokens to keep for top-k-filtering. Default to -100.
        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
        Returns:
        Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()
        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        x_len = x_lens.max()
        x_attn_mask = torch.zeros((x_len, x_len), dtype=torch.bool)

        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = torch.concat([x, y_pos], dim=1)

            y_len = y.shape[1]
            x_attn_mask_pad = F.pad(
                x_attn_mask,
                (0, y_len),
                value=True,
            )
            y_attn_mask = F.pad(
                torch.triu(
                    torch.ones(y_len, y_len, dtype=torch.bool), diagonal=1
                ),
                (x_len, 0),
                value=False,
            )
            xy_attn_mask = torch.concat(
                [x_attn_mask_pad, y_attn_mask], dim=0
            ).to(y.device)

            xy_dec, _ = self.ar_decoder(
                (xy_pos, None),
                mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"USLM EOS [{prompts.shape[1]} -> {y.shape[1]}]")
                break

            y = torch.concat([y, samples], dim=1)

        codes = [y[:, prefix_len + int(self.ar_audio_prepend_bos) :]]
        if self.num_quantizers == 1:
            return torch.stack(codes, dim=-1)

        # Non-AR Decoders
        y_emb = self.nar_audio_embeddings[0](
            y[:, int(self.ar_audio_prepend_bos) :]
        )

        if self.prefix_mode in [2, 4]:  # Exclude enrolled_phonemes
            enrolled_len = enroll_x_lens.max().item()
            # SOS + Synthesis Text + EOS
            text = torch.concat(
                [
                    text[:, :1],
                    text[:, enrolled_len - 1 :],
                ],
                dim=1,
            )
            text_len = text_len - (enrolled_len - 2)
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, self.num_quantizers):
                y_emb[:, :prefix_len] += self.nar_audio_embeddings[j](
                    prompts[..., j]
                )

            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_prenet(y_emb)
                y_pos = self.nar_audio_position(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < self.num_quantizers - 2:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)