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

# Protect the inference stack, not every package in the base image.
# In particular, do not freeze unrelated Ubuntu Python packages.
RUN python3 - <<'PY'
from importlib.metadata import distributions

def normalize(name):
    return name.lower().replace("_", "-").replace(".", "-")

protected_names = {
    "vllm",
    "torch",
    "torchvision",
    "torchaudio",
    "torchcodec",
    "triton",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "numpy",
    "compressed-tensors",
    "xgrammar",
    "outlines-core",
    "llguidance",
    "mistral-common",
    "cuda-python",
    "cuda-bindings",
    "cuda-pathfinder",
    "apache-tvm-ffi",
    "tvm-ffi",
}

requirements = set()

for dist in distributions():
    raw_name = dist.metadata.get("Name")
    if not raw_name:
        continue

    name = normalize(raw_name)

    if (
        name in protected_names
        or name.startswith("nvidia-")
        or name.startswith("flashinfer")
    ):
        requirements.add(f"{name}=={dist.version}")

with open("/opt/inference-constraints.txt", "w") as f:
    f.write("\n".join(sorted(requirements)) + "\n")

print("Protected inference packages:")
print(open("/opt/inference-constraints.txt").read())
PY

# Require modern application packages.
# Wheels only: fail clearly rather than compiling obsolete releases.
RUN uv pip install \
        --python "$(command -v python3)" \
        --no-cache \
        --only-binary :all: \
        --constraint /opt/inference-constraints.txt \
        "jupyterlab>=4" \
        "ipykernel>=6" \
        pytz \
        rapidfuzz \
        "pandas>=2.2,<3" \
        geopy \
        "accelerate>=1" \
        gguf \
        textstat \
        "spacy>=3.8,<3.9" \
        "textacy>=0.13,<0.14" \
        "nltk>=3.9" \
        "scikit-learn>=1.4" \
        "scipy>=1.12" \
        "sentence-transformers>=5" \
        timm \
        requests \
        tqdm \
        openai \
        anthropic \
        xai-sdk

# Keep this separate so any model-download failure is easy to identify.
# Apply the same inference constraints to pip invoked by spaCy.
RUN PIP_CONSTRAINT=/opt/inference-constraints.txt \
    PIP_ONLY_BINARY=:all: \
    python3 -m spacy download en_core_web_sm

RUN python3 -m nltk.downloader \
    -d /usr/local/share/nltk_data \
    words

# Confirm protected package versions were not changed.
# Test CPU-side imports without requiring a GPU on the build runner.
RUN python3 - <<'PY'
from importlib.metadata import version

with open("/opt/inference-constraints.txt") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        name, expected = line.split("==", 1)
        actual = version(name)

        if actual != expected:
            raise RuntimeError(
                f"Inference package changed: "
                f"{name}: expected {expected}, found {actual}"
            )

import torch
import pandas
import scipy
import sklearn
import spacy
import textacy
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
doc = nlp("This is a short image-build test.")

if not list(doc.sents):
    raise RuntimeError("spaCy model did not produce sentences")

print("CPU-side import checks passed.")
print("vLLM:", version("vllm"))
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Transformers:", version("transformers"))
print("pandas:", pandas.__version__)
print("spaCy:", spacy.__version__)
PY

RUN uv pip freeze \
        --python "$(command -v python3)" \
        > /opt/installed-environment.txt

COPY start-jupyter.sh /usr/local/bin/start-jupyter.sh

RUN sed -i 's/\r$//' /usr/local/bin/start-jupyter.sh \
    && chmod +x /usr/local/bin/start-jupyter.sh \
    && mkdir -p /workspace

WORKDIR /workspace

EXPOSE 8888

ENTRYPOINT []
CMD ["/usr/local/bin/start-jupyter.sh"]
