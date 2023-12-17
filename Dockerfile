ARG PYTORCH_STACK_VERSION=latest
FROM wallart/dl_pytorch:${PYTORCH_STACK_VERSION}
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
RUN mv hyperion_tmp/hyperion_server.py /usr/bin/hyperion_server
RUN mv hyperion_tmp/memory_server.py /usr/bin/memory_server
RUN chmod +x /usr/bin/hyperion_server
RUN chmod +x /usr/bin/memory_server

RUN rm -rf hyperion_tmp/

RUN mkdir -p /root/.hyperion/resources
RUN mkdir /root/.hyperion/resources/keys
ADD resources/ssl /root/.hyperion/resources/ssl
#ADD resources/keys /root/.hyperion/resources/keys
ADD resources/prompts /root/.hyperion/resources/prompts
ADD resources/gpt_models.json /root/.hyperion/resources/gpt_models.json
ADD resources/speakers_samples /root/.hyperion/resources/speakers_samples
ADD resources/voices_samples /root/.hyperion/resources/voices_samples
ADD resources/default_sentences /root/.hyperion/resources/default_sentences

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

#ENTRYPOINT ["/usr/bin/hyperion_server"]
#CMD ["--foreground", "restart", "--port", "6450"]

# Using HEREDOC notation
RUN <<EOF cat > /usr/sbin/bootstrap
#!/bin/bash
exec /usr/local/bin/runsvdir -P /etc/service
EOF
RUN chmod 755 /usr/sbin/bootstrap

ENTRYPOINT ["/usr/sbin/bootstrap"]