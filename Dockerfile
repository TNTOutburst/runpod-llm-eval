# syntax=docker/dockerfile:1

FROM vllm/vllm-openai:nightly

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=4 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        make \
        cmake \
        ninja-build \
        curl \
        wget \
        aria2 \
        zip \
        unzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Constrain packages already provided by the upstream image.
# Adding evaluation packages must not silently replace the inference stack.
RUN python3 - <<'PY'
from importlib.metadata import distributions

requirements = sorted({
    f"{dist.metadata['Name']}=={dist.version}"
    for dist in distributions()
    if dist.metadata["Name"]
})

with open("/opt/base-constraints.txt", "w") as f:
    f.write("\n".join(requirements) + "\n")
PY

RUN uv pip install \
        --python "$(command -v python3)" \
        --no-cache \
        --constraint /opt/base-constraints.txt \
        jupyterlab \
        ipykernel \
        pytz \
        rapidfuzz \
        pandas \
        geopy \
        accelerate \
        gguf \
        textstat \
        spacy \
        textacy \
        nltk \
        scikit-learn \
        scipy \
        sentence-transformers \
        timm \
        requests \
        tqdm \
        openai \
        anthropic \
        xai-sdk \
    && PIP_CONSTRAINT=/opt/base-constraints.txt \
        python3 -m spacy download en_core_web_sm

# Small CPU-side resources used by the evaluation application.
RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data words

# Record the exact installed environment for debugging/reproducibility.
RUN uv pip freeze \
        --python "$(command -v python3)" \
        > /opt/installed-environment.txt

COPY start-jupyter.sh /usr/local/bin/start-jupyter.sh

RUN sed -i 's/\r$//' /usr/local/bin/start-jupyter.sh \
    && chmod +x /usr/local/bin/start-jupyter.sh \
    && mkdir -p /workspace

WORKDIR /workspace

EXPOSE 8888

# Clear the upstream API-server entrypoint.
ENTRYPOINT []
CMD ["/usr/local/bin/start-jupyter.sh"]
