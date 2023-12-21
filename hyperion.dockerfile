ARG PYTORCH_STACK_VERSION=latest
ARG LLAMA_CPP_VERSION=latest
FROM wallart/dl_pytorch:${PYTORCH_STACK_VERSION} AS base
FROM wallart/llama_cpp:${LLAMA_CPP_VERSION} AS llama

FROM base
LABEL Author='Julien WALLART'

EXPOSE 6450
WORKDIR /tmp
SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt install -y libportaudio2 libsndfile1-dev python3-tk ffmpeg

RUN mkdir hyperion_tmp
ADD hyperion/ hyperion_tmp/hyperion
ADD hyperion_server.py hyperion_tmp/.
ADD memory_server.py hyperion_tmp/.
ADD requirements.txt hyperion_tmp/.
ADD setup.py hyperion_tmp/.

RUN cd hyperion_tmp; pip install .
# Temporary workaround to bypass conflicts
RUN pip install TTS==0.22.0
RUN git clone https://github.com/salesforce/LAVIS
RUN cd LAVIS; git checkout ac8fc98; \
    sed -i '38s/open3d==0.13.0/open3d==0.17.0/' requirements.txt; \
    sed -i '24d' lavis/models/blip_models/blip.py; \
    sed -i '23d' lavis/models/blip_models/blip.py
RUN cd LAVIS; pip install .
RUN rm -rf LAVIS
RUN mv hyperion_tmp/hyperion_server.py /usr/bin/hyperion_server
RUN mv hyperion_tmp/memory_server.py /usr/bin/memory_server
RUN chmod +x /usr/bin/hyperion_server
RUN chmod +x /usr/bin/memory_server

RUN rm -rf hyperion_tmp/

RUN mkdir -p /root/.hyperion/resources
RUN mkdir /root/.hyperion/resources/keys
RUN mkdir /root/.hyperion/resources/secret
ADD resources/ssl /root/.hyperion/resources/ssl
#ADD resources/keys /root/.hyperion/resources/keys
ADD resources/prompts /root/.hyperion/resources/prompts
ADD resources/llm_models.json /root/.hyperion/resources/llm_models.json
ADD resources/speakers_samples /root/.hyperion/resources/speakers_samples
ADD resources/voices_samples /root/.hyperion/resources/voices_samples
ADD resources/default_sentences /root/.hyperion/resources/default_sentences
ADD resources/secret/private_key.pem /root/.hyperion/resources/secret/private_key.pem

RUN mkdir -p /etc/service/memory_server/
RUN <<EOF cat > /etc/service/memory_server/run
#!/bin/bash
/usr/bin/memory_server
EOF
RUN chmod 755 /etc/service/memory_server/run

RUN mkdir -p /etc/service/hyperion_server/
RUN <<EOF cat > /etc/service/hyperion_server/run
#!/bin/bash
name_opt=""
if [[ -n \$NAME ]]
then
    name_opt="--name \$NAME"
fi
/usr/bin/hyperion_server --foreground restart --port 6450 \$name_opt
EOF
RUN chmod 755 /etc/service/hyperion_server/run

RUN mkdir -p /etc/service/llama_cpp_server/
RUN <<EOF cat > /etc/service/llama_cpp_server/run
#!/bin/bash
host=0.0.0.0
port=8080
threads=\$(nproc --all)
parallel_requests=1
num_layers=33
tokens=8192
model_dir=\$(ls -I llama.cpp /root)
model_path=/root/\$model_dir/ggml-model-q4_0.gguf
opts="-m \$model_path --embedding --host \$host --port \$port -ngl \$num_layers -t \$threads -np \$parallel_requests"
/root/llama.cpp/server -c \$tokens \$opts
EOF
RUN chmod 755 /etc/service/llama_cpp_server/run

# Using HEREDOC notation
RUN <<EOF cat > /usr/sbin/bootstrap
#!/bin/bash
exec /usr/local/bin/runsvdir -P /etc/service
EOF
RUN chmod 755 /usr/sbin/bootstrap

COPY --from=llama /root /root

ENTRYPOINT ["/usr/sbin/bootstrap"]