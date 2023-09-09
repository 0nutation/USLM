
audio_extractor="SpeechTokenizer"

st_dir="ckpt/speechtokenizer/"
uslm_dir="ckpt/uslm/"
out_dir="output/"

mkdir -p ${st_dir}
mkdir -p ${uslm_dir}
mkdir -p ${out_dir}

if [ ! -e "${st_dir}/config.json" ];then
    cd ${st_dir}
    wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/SpeechTokenizer.pt"
    wget "https://huggingface.co/fnlp/SpeechTokenizer/resolve/main/speechtokenizer_hubert_avg/config.json" 
    cd -
fi 

if [ ! -e "${uslm_dir}/USLM.pt" ];then
    cd ${uslm_dir}
    wget "https://huggingface.co/fnlp/USLM/resolve/main/USLM_libritts/USLM.pt"
    wget "https://huggingface.co/fnlp/USLM/resolve/main/USLM_libritts/unique_text_tokens.k2symbols" 
    cd -
fi 


python3 bin/infer.py --output-dir ${out_dir}/ \
    --model-name uslm --norm-first true --add-prenet false \
    --share-embedding true --norm-first true --add-prenet false \
    --audio-extractor "${audio_extractor}" \
    --speechtokenizer-dir "${st_dir}" \
    --checkpoint=${uslm_dir}/best-valid-loss.pt \
    --text-tokens "${uslm_dir}/unique_text_tokens.k2symbols" \
    --text-prompts "mr Soames was a tall, spare man, of a nervous and excitable temperament." \
    --audio-prompts prompts/1580_141083_000002_000002.wav \
    --text "Begin with the fundamental steps of the process. This will give you a solid foundation to build upon and boost your confidence. " \
    
