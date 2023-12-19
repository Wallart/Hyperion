ARG PYTORCH_STACK_VERSION=latest
FROM wallart/dl_pytorch:${PYTORCH_STACK_VERSION}
LABEL Author='Julien WALLART'

# ARG must be used on the line right after, before being lost
ARG MODEL
ENV MISTRAL_MODEL=${MODEL:-Mixtral-8x7B-Instruct-v0.1}
ARG ALIAS
ENV MISTRAL_ALIAS=${ALIAS:-mixtral-8x7B}
ARG TOKENS
ENV MISTRAL_TOKENS=${TOKENS:-32000}

WORKDIR /root

RUN apt install git-lfs
RUN git lfs install
RUN git clone https://huggingface.co/mistralai/$MISTRAL_MODEL

RUN git clone https://github.com/ggerganov/llama.cpp
RUN cd llama.cpp; make
RUN cd llama.cpp; pip install -r requirements.txt

RUN python llama.cpp/convert.py $MISTRAL_MODEL
RUN ./llama.cpp/quantize $MISTRAL_MODEL/ggml-model-f16.gguf $MISTRAL_MODEL/ggml-model-q4_0.gguf q4_0
RUN cp $MISTRAL_MODEL/ggml-model-q4_0.gguf .; rm -rf $MISTRAL_MODEL; \
    mkdir $MISTRAL_MODEL; mv ggml-model-q4_0.gguf $MISTRAL_MODEL/.