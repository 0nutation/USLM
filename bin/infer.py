"""
Created on Fri. Sept. 8 00:43:42 2023
@author: Dong Zhang
"""

import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torchaudio

from uslm.data import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text,
    Speechtokenizer,
    sttokenize_audio
)
from uslm.data.collation import get_text_token_collater
from uslm.models import add_model_arguments, get_model


def circular_padding(x, tgt_len):
    if x.size(-1) >= tgt_len:
        return x[:, :, :tgt_len]
    t = tgt_len // x.size(-1)
    r = tgt_len % x.size(-1)
    tgt = x.repeat(1, 1, t)
    tgt = torch.cat([tgt, x[:, :, :r]], axis=-1)
    return tgt


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text-prompts",
        type=str,
        default="",
        help="Text prompts which are separated by |.",
    )

    parser.add_argument(
        "--audio-prompts",
        type=str,
        default="",
        help="Audio prompts which are separated by | and should be aligned with --text-prompts.",
    )

    parser.add_argument(
        "--text",
        type=str,
        default="To get up and running quickly just follow the steps below.",
        help="Text to be synthesized.",
    )

    parser.add_argument(
        "--audio-extractor",
        type=str,
        default="Encodec",
        help="Encodec or SpeechTokenizer or Fbank",
    )

    # model
    add_model_arguments(parser)

    parser.add_argument(
        "--text-tokens",
        type=str,
        default="data/tokenized/unique_text_tokens.k2symbols",
        help="Path to the unique text tokens file.",
    )
    parser.add_argument(
        "--text-extractor",
        type=str,
        default="espeak",
        help="espeak or pypinyin or pypinyin_initials_finals",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="exp/vallf_nano_full/checkpoint-100000.pt",
        help="Path to the saved checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("infer/demo"),
        help="Path to the tokenized files.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=-100,
        help="Whether AR Decoder do top_k(if > 0) sampling.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The temperature of AR Decoder top_k sampling.",
    )

    parser.add_argument(
        "--without-nar",
        default=False,
        help="without nar",
    )
    parser.add_argument(
        "--speechtokenizer-dir",
        type=str,
        default="False",
        help="dirname of speechtokenizer models",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    text_tokenizer = TextTokenizer(backend=args.text_extractor)
    text_collater = get_text_token_collater(args.text_tokens)

    if args.audio_extractor == "EnCodec":
        sr = 24000
        audio_tokenizer = AudioTokenizer()
        tokenize_a = tokenize_audio
    elif args.audio_extractor == "SpeechTokenizer":
        sr = 16000
        audio_tokenizer = Speechtokenizer(ckpt_dir=args.speechtokenizer_dir)
        tokenize_a = sttokenize_audio


    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)

    model = get_model(args)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model"], strict=True
        )
        assert not missing_keys


    model.to(device)
    model.eval()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if os.path.isfile(args.text):  # for demos

        with open(args.text) as f:
            for line in tqdm(f):
                fields = line.strip().split("|")
                assert len(fields) == 4
                prompt_text, prompt_audio, text, audio_path = fields
                logging.info(f"synthesize text: {text}")
                os.makedirs(os.path.dirname(audio_path),exist_ok=True)
                text_tokens, text_tokens_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text} {text}".strip()
                        )
                    ]
                )
                _, enroll_x_lens = text_collater(
                    [
                        tokenize_text(
                            text_tokenizer, text=f"{prompt_text}".strip()
                        )
                    ]
                )

                audio_prompts = tokenize_a(audio_tokenizer, prompt_audio)
                if args.audio_extractor == "SpeechTokenizer":
                    audio_prompts = audio_prompts.permute(1, 2, 0).to(device) #[b,t,8]
                elif args.audio_extractor == "EnCodec":
                    audio_prompts = audio_prompts[0][0].transpose(2,1)

                # synthesis
                encoded_frames = model.inference(
                    text_tokens.to(device),
                    text_tokens_lens.to(device),
                    audio_prompts,
                    enroll_x_lens=enroll_x_lens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )

                if args.audio_extractor == "SpeechTokenizer":
                    code_generated = encoded_frames.permute(2,0,1) #[8,b,T]
                else:
                    code_generated = [(encoded_frames.transpose(2, 1), None)]

                
                if args.without_nar:
                    audio_prompts = circular_padding(audio_prompts.permute(2,0,1), code_generated.shape[-1])
                    code_generated = torch.cat((code_generated[:1,:,:], audio_prompts[1:4,:,:]),dim=0)
                    
                samples = audio_tokenizer.decode(
                    code_generated
                )
                # store
                torchaudio.save(audio_path, samples[0].cpu(), sr)
        return




    text_prompts = " ".join(args.text_prompts.split("|"))

    audio_prompts = []
    if args.audio_prompts:
        for n, audio_file in enumerate(args.audio_prompts.split("|")):
            encoded_frames = tokenize_a(audio_tokenizer, audio_file)
            if False:
                samples = audio_tokenizer.decode(encoded_frames)
                torchaudio.save(
                    f"{args.output_dir}/p{n}.wav", samples[0], sr
                )

            if args.audio_extractor == "EnCodec":
                audio_prompts.append(encoded_frames[0][0])
            elif args.audio_extractor == "SpeechTokenizer":
                audio_prompts.append(encoded_frames.permute(1,0,2))


        assert len(args.text_prompts.split("|")) == len(audio_prompts)
        audio_prompts = torch.concat(audio_prompts, dim=-1).transpose(2, 1)
        audio_prompts = audio_prompts.to(device)

    

    for n, text in enumerate(args.text.split("|")):
        logging.info(f"synthesize text: {text}")
        text_tokens, text_tokens_lens = text_collater(
            [
                tokenize_text(
                    text_tokenizer, text=f"{text_prompts} {text}".strip()
                )
            ]
        )

        # synthesis

        enroll_x_lens = None
        if text_prompts:
            _, enroll_x_lens = text_collater(
                [
                    tokenize_text(
                        text_tokenizer, text=f"{text_prompts}".strip()
                    )
                ]
            )
        encoded_frames = model.inference(
            text_tokens.to(device),
            text_tokens_lens.to(device),
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=args.top_k,
            temperature=args.temperature,
        )


        if audio_prompts != []:
            if args.audio_extractor == "SpeechTokenizer":
                code_generated = encoded_frames.permute(2,0,1)
            else:
                code_generated = [(encoded_frames.transpose(2, 1), None)]
            samples = audio_tokenizer.decode(
                code_generated
            )
            # store
            idx = args.audio_prompts.split('|')[n]
            torchaudio.save(
                f"{args.output_dir}/gen-{os.path.basename(idx).replace('flac','wav')}", samples[0].cpu(), sr
            )
        else:  # Transformer
            pass


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    main()
