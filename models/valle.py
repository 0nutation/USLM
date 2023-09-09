# Copyright    2023                             (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

from .macros import NUM_AUDIO_TOKENS, NUM_TEXT_TOKENS
from .visualizer import visualize


class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)


# NOTE: There are two ways to implement the model
#       1) [VALL-F] standard TransformerDecoder, use x as memory
#       2) [VALL-E] modified TransformerDecoder like GPT-x(e.g. causal TransformerEncoder),
#          use x as the prefix of decoder inputs
class VALLF(nn.Module):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        norm_first: bool = True,
        add_prenet: bool = False,
        decoder_cls: Union[
            nn.TransformerDecoder, nn.TransformerEncoder
        ] = nn.TransformerDecoder,
        decoder_layer_cls: Union[
            TransformerDecoderLayer, TransformerEncoderLayer
        ] = TransformerDecoderLayer,
        prefix_mode: int = 0,
        share_embedding: bool = True,
        nar_scale_factor: float = 1.0,
        prepend_bos: bool = False,
        num_quantizers: int = 8,
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
        super().__init__()
        nar_d_model = int(d_model * nar_scale_factor)

        self.ar_text_embedding = TokenEmbedding(d_model, NUM_TEXT_TOKENS)  # W_x
        self.nar_text_embedding = TokenEmbedding(nar_d_model, NUM_TEXT_TOKENS)

        # ID NUM_AUDIO_TOKENS     -> PAD
        # ID NUM_AUDIO_TOKENS + 1 -> BOS
        self.ar_audio_prepend_bos = prepend_bos
        self.ar_audio_embedding = TokenEmbedding(
            d_model, NUM_AUDIO_TOKENS + 1 + int(prepend_bos)
        )

        # PreNet
        if add_prenet:
            self.ar_text_prenet = nn.Sequential(
                Transpose(),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Conv1d(d_model, d_model, kernel_size=5, padding="same"),
                nn.BatchNorm1d(d_model),
                nn.ReLU(),
                nn.Dropout(0.5),
                Transpose(),
                nn.Linear(d_model, d_model),
            )

            self.ar_audio_prenet = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, d_model),
            )
        else:
            self.ar_text_prenet = nn.Identity()
            self.ar_audio_prenet = nn.Identity()

        self.ar_text_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )
        self.ar_audio_position = SinePositionalEmbedding(
            d_model,
            dropout=0.1,
            scale=False,
            alpha=True,
        )

        self.ar_decoder = decoder_cls(
            decoder_layer_cls(
                d_model,
                nhead,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=num_layers,
            norm=LayerNorm(d_model) if norm_first else None,
        )
        self.ar_predict_layer = nn.Linear(
            d_model, NUM_AUDIO_TOKENS + 1, bias=False
        )

        self.ar_accuracy_metric = MulticlassAccuracy(
            NUM_AUDIO_TOKENS + 1,
            top_k=10,
            average="micro",
            multidim_average="global",
            ignore_index=NUM_AUDIO_TOKENS,
        )

        self.rng = random.Random(0)
        self.num_heads = nhead
        self.prefix_mode = prefix_mode
        self.num_quantizers = num_quantizers

        assert num_quantizers >= 1
        if num_quantizers > 1:
            self.nar_audio_embeddings = nn.ModuleList(
                [TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS + 1)]
                + [
                    TokenEmbedding(nar_d_model, NUM_AUDIO_TOKENS)
                    for i in range(num_quantizers - 1)
                ]
            )  # W_a

            # PreNet
            if add_prenet:
                self.nar_text_prenet = nn.Sequential(
                    Transpose(),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Conv1d(
                        nar_d_model, nar_d_model, kernel_size=5, padding="same"
                    ),
                    nn.BatchNorm1d(nar_d_model),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    Transpose(),
                    nn.Linear(nar_d_model, nar_d_model),
                )
                self.nar_audio_prenet = nn.Sequential(
                    nn.Linear(nar_d_model, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(0.25),
                    nn.Linear(256, nar_d_model),
                )
            else:
                self.nar_text_prenet = nn.Identity()
                self.nar_audio_prenet = nn.Identity()

            self.nar_text_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.0,
                scale=False,
                alpha=False,
            )
            self.nar_audio_position = SinePositionalEmbedding(
                nar_d_model,
                dropout=0.1,
                scale=False,
                alpha=False,
            )

            self.nar_decoder = decoder_cls(
                decoder_layer_cls(
                    nar_d_model,
                    int(nhead * nar_scale_factor),
                    dim_feedforward=nar_d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                    norm_first=norm_first,
                    adaptive_layer_norm=True,
                ),
                num_layers=int(num_layers * nar_scale_factor),
                norm=AdaptiveLayerNorm(
                    nar_d_model, norm=nn.LayerNorm(nar_d_model)
                )
                if norm_first
                else None,
            )
            self.nar_predict_layers = nn.ModuleList(
                [
                    nn.Linear(nar_d_model, NUM_AUDIO_TOKENS, bias=False)
                    for i in range(num_quantizers - 1)
                ]
            )
            self.nar_stage_embeddings = nn.ModuleList(
                [
                    TokenEmbedding(nar_d_model, 1)
                    for i in range(num_quantizers - 1)
                ]
            )

            if share_embedding:
                # We share the parameters of the output projection layer with the parameters of the acoustic embedding Wa
                # NOTE(Feiteng): In the experiment, this undermines accuracy
                # self.ar_predict_layer.weight = self.ar_audio_embedding.weight

                # We also share the parameters of the acoustic embedding layer and the output prediction layer,
                # which means the weights of the j-th prediction layer are the same as the (j + 1)-th acoustic embedding layer.
                for j in range(0, num_quantizers - 2):
                    self.nar_predict_layers[
                        j
                    ].weight = self.nar_audio_embeddings[j + 2].weight

            self.nar_accuracy_metric = MulticlassAccuracy(
                NUM_AUDIO_TOKENS + 1,
                top_k=10,
                average="micro",
                multidim_average="global",
                ignore_index=NUM_AUDIO_TOKENS,
            )

    #     self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     if isinstance(module, (nn.Linear)):
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #         if isinstance(module, nn.Linear) and module.bias is not None:
    #             module.bias.data.zero_()
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     elif isinstance(module, nn.Embedding):
    #         module.weight.data.normal_(mean=0.0, std=1.0)

    def stage_parameters(self, stage: int = 1) -> Iterator[nn.Parameter]:
        assert stage > 0
        if stage == 1:
            for name, param in self.named_parameters():
                if name.startswith("ar_"):
                    print(f" AR parameter: {name}")
                    yield param

        if stage == 2:
            for name, param in self.named_parameters():
                if name.startswith("nar_"):
                    print(f"NAR parameter: {name}")
                    yield param

    def stage_named_parameters(
        self, stage: int = 1
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        assert stage > 0
        if stage == 1:
            for pair in self.named_parameters():
                if pair[0].startswith("ar_"):
                    yield pair

        if stage == 2:
            for pair in self.named_parameters():
                if pair[0].startswith("nar_"):
                    yield pair

    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = F.pad(y, (0, 1), value=0) + eos_id * F.pad(
            y_mask_int, (0, 1), value=1
        )
        # inputs, targets
        if self.ar_audio_prepend_bos:
            return (
                F.pad(targets[:, :-1], (1, 0), value=NUM_AUDIO_TOKENS + 1),
                targets,
            )

        return targets[:, :-1], targets[:, 1:]

    def _prepare_prompts(self, y, y_lens, codes, nar_stage, y_prompts_codes, ):
        # 5.1 For the NAR acoustic prompt tokens, we select a random segment waveform of 3 seconds
        # from the same utterance.
        # We implement this differently.
        if self.prefix_mode == 0:
            # no prefix
            prefix_len = 0
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, nar_stage):
                # Formula (4) (5)
                y_emb = y_emb + self.nar_audio_embeddings[j](codes[..., j])
        elif self.prefix_mode == 1:
            # prefix at begining
            int_low = (0.25 * y_lens.min()).type(torch.int64).item()
            prefix_len = torch.randint(int_low, int_low * 2, size=()).item()
            prefix_len = min(prefix_len, 150)  # 16000/320 * 3s = 150 frames


            y_prompts = self.nar_audio_embeddings[0](y[:, :prefix_len])
            y_emb = self.nar_audio_embeddings[0](y[:, prefix_len:])
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    codes[:, :prefix_len, j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](
                        codes[:, prefix_len:, j]
                    )
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        elif self.prefix_mode in [2, 4]:
            if self.prefix_mode == 2:
                # random prefix
                prefix_len = min(150, int(0.25 * y_lens.min().item()))

                y_prompts_codes = []
                for b in range(codes.shape[0]):
                    start = self.rng.randint(0, y_lens[b].item() - prefix_len)
                    y_prompts_codes.append(
                        torch.clone(codes[b, start : start + prefix_len])
                    )
                    codes[
                        b, start : start + prefix_len, nar_stage
                    ] = NUM_AUDIO_TOKENS
                y_prompts_codes = torch.stack(y_prompts_codes, dim=0)
            else:
                prefix_len = y_prompts_codes.shape[1]

            y_prompts = self.nar_audio_embeddings[0](y_prompts_codes[..., 0])
            y_emb = self.nar_audio_embeddings[0](y)
            for j in range(1, self.num_quantizers):
                y_prompts += self.nar_audio_embeddings[j](
                    y_prompts_codes[..., j]
                )
                if j < nar_stage:
                    y_emb += self.nar_audio_embeddings[j](codes[..., j])
            y_emb = torch.concat([y_prompts, y_emb], axis=1)
        else:
            raise ValueError

        return y_emb, prefix_len

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

        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        total_loss, metrics = 0.0, {}

        y_mask = make_pad_mask(y_lens).to(y.device)
        y_mask_int = y_mask.type(torch.int64)

        codes = y.type(torch.int64) * (1 - y_mask_int.unsqueeze(dim=-1))

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(
            codes[..., 0], y_mask_int, eos_id=NUM_AUDIO_TOKENS
        )

        if train_stage in [0, 1]:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            ar_y_mask = y_mask
            if self.ar_audio_prepend_bos:
                ar_y_mask = F.pad(y_mask, (1, 0), value=False)

            y_len = y_lens.max() + int(self.ar_audio_prepend_bos)
            tgt_mask = torch.triu(
                torch.ones(y_len, y_len, device=y.device, dtype=torch.bool),
                diagonal=1,
            )
            y_dec, _ = self.ar_decoder(
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=ar_y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.ar_predict_layer(y_dec).permute(0, 2, 1)
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
            nar_stage = self.rng.choices(
                [_k for _k in range(1, self.num_quantizers)],
                weights=[0.5] + [0.5 / (num_nar_layers-1)] * (num_nar_layers-1),  # weights=[1.0 / num_nar_layers] * num_nar_layers,
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
                targets = targets
                y_mask = F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False)
            elif self.prefix_mode == 1:
                targets = targets[:, prefix_len:]
            else:
                assert prefix_len == 0

            y_pos = self.nar_audio_prenet(y_emb)
            y_pos = self.nar_audio_position(y_pos)

            y_dec, _ = self.nar_decoder(
                (y_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
                x,
                tgt_mask=None,
                tgt_key_padding_mask=y_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            if self.prefix_mode != 0:
                y_dec = y_dec[:, prefix_len:]
                if self.prefix_mode == 4:
                    prefix_len = 0  # reset for Top10Accuracy metric

            logits = self.nar_predict_layers[nar_stage - 1](y_dec).permute(
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
            metrics["NarTop10Accuracy"] = (
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
        enroll_x_lens: Union[torch.Tensor, None] = None,
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
          Return the predicted audio code matrix and cross-entropy loss.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)

        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)
        # NOTE: x has been padded in TextTokenCollater
        x_mask = make_pad_mask(x_lens).to(x.device)

        prompts = y
        prefix_len = y.shape[1]

        # AR Decoder
        # TODO: Managing decoder steps avoid repetitive computation
        y = prompts[..., 0]
        if self.ar_audio_prepend_bos:
            y = F.pad(y, (1, 0), value=NUM_AUDIO_TOKENS + 1)

        while True:
            y_emb = self.ar_audio_embedding(y)
            y_emb = self.ar_audio_prenet(y_emb)
            y_pos = self.ar_audio_position(y_emb)

            tgt_mask = torch.triu(
                torch.ones(
                    y.shape[1], y.shape[1], device=y.device, dtype=torch.bool
                ),
                diagonal=1,
            )

            y_dec, _ = self.ar_decoder(
                (y_pos, None),
                x,
                tgt_mask=tgt_mask,
                memory_mask=None,
                memory_key_padding_mask=x_mask,
            )
            logits = self.ar_predict_layer(y_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=1.0, temperature=temperature
            )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prefix_len) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"VALL-F EOS [{prefix_len} -> {y.shape[1]}]")
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
            assert text.shape[0] == 1

        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        if self.prefix_mode != 0:
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
            y_dec, _ = self.nar_decoder(
                (y_pos, self.nar_stage_embeddings[i].weight),
                x,
                tgt_mask=None,
                memory_mask=None,
                memory_key_padding_mask=None,
            )
            logits = predict_layer(y_dec[:, prefix_len:])
            samples = torch.argmax(logits, dim=-1)
            codes.append(samples)
            # Formula (4) (5)
            if i < 6:
                if self.prefix_mode == 0:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == self.num_quantizers
        return torch.stack(codes, dim=-1)

    def visualize(
        self,
        predicts: Tuple[torch.Tensor],
        batch: Dict[str, Union[List, torch.Tensor]],
        output_dir: str,
        limit: int = 4,
    ) -> None:
        visualize(predicts, batch, output_dir, limit=limit)


class VALLE(VALLF):
    """It implements https://arxiv.org/abs/2301.02111
    "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
    """

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
        super(VALLE, self).__init__(
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


            # # layer2-8
            # nar_stage = self.rng.choices(
            #     [_k for _k in range(1, self.num_quantizers)],
            #     weights=[1.0 / (num_nar_layers)] * (num_nar_layers),
            #     k=1,
            # )[0]

            # x = self.nar_text_embedding(text)
            # x = self.nar_text_prenet(x)
            # x = self.nar_text_position(x)

            # y_emb, prefix_len = self._prepare_prompts(
            #     y, y_lens, codes, nar_stage, y_prompts_codes
            # )

            # y_len = y_lens.max()
            # targets = codes[..., nar_stage] + NUM_AUDIO_TOKENS * y_mask_int
            # if self.prefix_mode in [2, 4]:
            #     xy_padding_mask = torch.concat(
            #         [
            #             x_mask,
            #             F.pad(y_mask, (y_emb.shape[1] - y_len, 0), value=False),
            #         ],
            #         dim=1,
            #     )
            # elif self.prefix_mode == 1:
            #     targets = targets[:, prefix_len:]

            # y_pos = self.nar_audio_prenet(y_emb)
            # y_pos = self.nar_audio_position(y_pos)
            # xy_pos = torch.concat([x, y_pos], dim=1)
            # xy_dec, _ = self.nar_decoder(
            #     (xy_pos, self.nar_stage_embeddings[nar_stage - 1].weight),
            #     src_key_padding_mask=xy_padding_mask,
            #     # is_causal=False,
            # )
            # xy_dec = xy_dec[:, x_lens.max() + prefix_len :]
            # if self.prefix_mode == 4:
            #     prefix_len = 0  # reset for Top10Accuracy metric
            # logits = self.nar_predict_layers[nar_stage - 1](xy_dec).permute(
            #     0, 2, 1
            # )

            # # loss
            # total_length = (y_lens).sum().type(torch.float32)
            # total_loss += (
            #     F.cross_entropy(
            #         logits,
            #         targets,
            #         ignore_index=NUM_AUDIO_TOKENS,
            #         reduction=reduction,
            #     )
            #     * (total_length / (total_length - prefix_len * x.shape[0]))
            # )
            # metrics["NarL2-8Top10Accuracy"] = (
            #     self.nar_accuracy_metric(
            #         F.pad(
            #             logits.detach(),
            #             (0, 0, 0, 1, 0, 0),
            #             value=logits.min().cpu().item(),
            #         ),
            #         targets,
            #     ).item()
            #     * total_length
            # )

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

            # max_repeated_tokens = 10

            # while y.shape[1]-prefix_len >= max_repeated_tokens and torch.all(samples[0, 0] == y[0, -max_repeated_tokens:]):
            #     samples = topk_sampling(
            #         logits, top_k=top_k, top_p=1.0, temperature=temperature
            #     )

            if (
                torch.argmax(logits, dim=-1)[0] == NUM_AUDIO_TOKENS
                or samples[0, 0] == NUM_AUDIO_TOKENS
                or (y.shape[1] - prompts.shape[1]) > x_lens.max() * 16
            ):
                if prompts.shape[1] == y.shape[1]:
                    raise SyntaxError(
                        "well trained model shouldn't reach here."
                    )

                print(f"VALL-E EOS [{prompts.shape[1]} -> {y.shape[1]}]")
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

    def continual(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
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
        Returns:
          Return the predicted audio code matrix.
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3, y.shape
        assert y.shape[0] == 1, y.shape

        assert torch.all(x_lens > 0)
        assert self.num_quantizers == 8

        # NOTE: x has been padded in TextTokenCollater
        text = x
        x = self.ar_text_embedding(text)
        x = self.ar_text_prenet(x)
        x = self.ar_text_position(x)

        text_len = x_lens.max()

        prefix_len = min(int(y.shape[1] * 0.5), 3 * 75)

        # AR Decoder
        prompts = y[:, :prefix_len]

        codes = [y[:, prefix_len:, 0]]
        # Non-AR Decoders
        x = self.nar_text_embedding(text)
        x = self.nar_text_prenet(x)
        x = self.nar_text_position(x)

        y_emb = self.nar_audio_embeddings[0](y[..., 0])

        if self.prefix_mode == 0:
            for i, (predict_layer, embedding_layer) in enumerate(
                zip(
                    self.nar_predict_layers,
                    self.nar_audio_embeddings[1:],
                )
            ):
                y_pos = self.nar_audio_position(y_emb)
                y_pos = self.nar_audio_prenet(y_pos)
                xy_pos = torch.concat([x, y_pos], dim=1)

                xy_dec, _ = self.nar_decoder(
                    (xy_pos, self.nar_stage_embeddings[i].weight)
                )
                logits = predict_layer(xy_dec[:, text_len + prefix_len :])

                samples = torch.argmax(logits, dim=-1)
                codes.append(samples)

                if i < 6:
                    y_emb[:, :prefix_len] += embedding_layer(
                        prompts[..., i + 1]
                    )
                    y_emb[:, prefix_len:] += embedding_layer(samples)
        else:
            for j in range(1, 8):
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

                if i < 6:
                    y_emb[:, prefix_len:] += embedding_layer(samples)

        assert len(codes) == 8
        return torch.stack(codes, dim=-1)


# https://github.com/microsoft/unilm/blob/master/xtune/src/transformers/modeling_utils.py
def top_k_top_p_filtering(
    logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(
            max(top_k, min_tokens_to_keep), logits.size(-1)
        )  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


def topk_sampling(logits, top_k=10, top_p=1.0, temperature=1.0):
    # temperature: (`optional`) float
    #     The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.
    # top_k: (`optional`) int
    #     The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.
    # top_p: (`optional`) float
    #     The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

    # Temperature (higher temperature => more likely to sample low probability tokens)
    if temperature != 1.0:
        logits = logits / temperature
    # Top-p/top-k filtering
    logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    # Sample
    token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    return token
