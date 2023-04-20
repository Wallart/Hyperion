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
ADD requirements.txt hyperion_tmp/.
ADD setup.py hyperion_tmp/.

RUN cd hyperion_tmp; pip install -r requirements.txt; python setup.py install
RUN mv hyperion_tmp/hyperion_server.py /usr/bin/hyperion_server
RUN chmod +x /usr/bin/hyperion_server

RUN rm -rf hyperion_tmp/

RUN mkdir -p /root/.hyperion/resources
ADD resources/keys /root/.hyperion/resources/keys
ADD resources/prompts /root/.hyperion/resources/prompts
ADD resources/speakers_samples /root/.hyperion/resources/speakers_samples
ADD resources/default_sentences /root/.hyperion/resources/default_sentences

ENTRYPOINT ["/usr/bin/hyperion_server"]
CMD ["--foreground", "restart", "--port", "6450"]