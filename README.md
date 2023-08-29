# USLM: Unified Speech Language Model

Build upon SpeechTokenizer, USLM consists of autoregressive and non-autoregressive models, it can hierarchically model information
in speech. The autoregressive (AR) model captures the content information by modeling tokens
from the first RVQ quantizer. The non-autoregressive (NAR) model complements paralinguistic
information for the AR model by generating tokens from the subsequent quantizers conditioned on
the first-layer tokens
