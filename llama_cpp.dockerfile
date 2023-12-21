# Global ARG, because we are before FROM
ARG PYTORCH_STACK_VERSION=latest
FROM wallart/dl_pytorch:${PYTORCH_STACK_VERSION}
LABEL Author='Julien WALLART'

# After a FROM, ARG is local to next build stage
ARG MODEL
ENV MISTRAL_MODEL=${MODEL:-Mixtral-8x7B-Instruct-v0.1}
ARG ALIAS
ENV MISTRAL_ALIAS=${ALIAS:-mixtral-8x7B}
ARG TOKENS
ENV MISTRAL_TOKENS=${TOKENS:-32000}

WORKDIR /root

RUN apt install git-lfs && \
  git lfs install && \
  git clone https://huggingface.co/mistralai/$MISTRAL_MODEL && \
  git clone https://github.com/ggerganov/llama.cpp && \
  cd llama.cpp && make LLAMA_CUBLAS=1 && pip install -r requirements.txt && \
  python ./convert.py /root/$MISTRAL_MODEL && \
  ./quantize /root/$MISTRAL_MODEL/ggml-model-f16.gguf /root/$MISTRAL_MODEL/ggml-model-q4_0.gguf q4_0 && \
  cp /root/$MISTRAL_MODEL/ggml-model-q4_0.gguf /root/.; \
  rm -rf /root/$MISTRAL_MODEL; \
  mkdir /root/$MISTRAL_MODEL; \
  mv /root/ggml-model-q4_0.gguf /root/$MISTRAL_MODEL/.