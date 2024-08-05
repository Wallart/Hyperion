# Global ARG, because we are before FROM
ARG PYTORCH_STACK_VERSION=latest
FROM wallart/dl_pytorch:${PYTORCH_STACK_VERSION}
LABEL Author='Julien WALLART'

# After a FROM, ARG is local to next build stage
ARG CUDA_ARCH=compute_86
ENV CUDA_DOCKER_ARCH=$CUDA_ARCH

ARG MODEL_PREFIX=mistralai
ARG MODEL=Mistral-7B-Instruct-v0.2
ENV LLM_MODEL=$MODEL

ARG ALIAS=mistral-7B
ENV LLM_ALIAS=$ALIAS

ARG TOKENS=8192
ENV LLM_TOKENS=$TOKENS

ARG LAYERS=33
ENV LLM_LAYERS=$LAYERS

ARG PAR_REQUESTS=2
ENV PARALLEL_REQUESTS=$PAR_REQUESTS

WORKDIR /root

RUN apt install git-lfs && \
  git lfs install && \
  mkdir models; cd models; git clone https://huggingface.co/$MODEL_PREFIX/$LLM_MODEL; cd .. && \
  git clone https://github.com/ggerganov/llama.cpp; cd llama.cpp && \
  sed -i 's/svr.Post("\/embedding"/svr.Post("\/embeddings"/' examples/server/server.cpp && \
  make LLAMA_CUBLAS=1 CUDA_DOCKER_ARCH=$CUDA_DOCKER_ARCH && pip install -r requirements.txt && \
  python ./convert.py /root/models/$LLM_MODEL && \
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH && \
  ./quantize /root/models/$LLM_MODEL/ggml-model-f16.gguf /root/models/$LLM_MODEL/ggml-model-q4_0.gguf q4_0 && \
  cp /root/models/$LLM_MODEL/ggml-model-q4_0.gguf /root/.; \
  rm -rf /root/models/$LLM_MODEL; \
  mkdir /root/models/$LLM_MODEL; \
  mv /root/ggml-model-q4_0.gguf /root/models/$LLM_MODEL/.

#RUN pip install uvicorn anyio starlette sse_starlette starlette_context fastapi pydantic_settings; \
#    CMAKE_ARGS="-DLLAMA_CUBLAS=ON -DCUDA_DOCKER_ARCH=$CUDA_DOCKER_ARCH" pip install llama-cpp-python
RUN mkdir -p /etc/service/llama_cpp_server/
RUN <<EOF cat > /etc/service/llama_cpp_server/run
#!/bin/bash
host=0.0.0.0
port=8080
threads=\$(nproc --all)
model_dir=\$(ls /root/models | xargs | cut -d ' ' -f 1)
model_path=/root/models/\$model_dir/ggml-model-q4_0.gguf
opts="-m \$model_path --embedding --host \$host --port \$port -t \$threads"
/root/llama.cpp/server -a \$LLM_ALIAS -c \$LLM_TOKENS -ngl \$LLM_LAYERS -np \$PARALLEL_REQUESTS \$opts
#opts="--model \$model_path --host \$host --port \$port --n_threads \$threads"
#python -m llama_cpp.server \$opts --model_alias \$LLM_ALIAS --n_gpu_layers \$LLM_LAYERS --n_ctx \$LLM_TOKENS --verbose true
EOF
RUN chmod 755 /etc/service/llama_cpp_server/run

RUN <<EOF cat > /usr/sbin/bootstrap
#!/bin/bash
exec /usr/local/bin/runsvdir -P /etc/service
EOF
RUN chmod 755 /usr/sbin/bootstrap

ENTRYPOINT ["/usr/sbin/bootstrap"]