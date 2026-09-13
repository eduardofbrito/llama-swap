# =============================================================================
# vllm-swap — llama-swap (este fork) sobre vLLM
#
# O config.yaml usa --load-format fastsafetensors, que exige o pacote
# fastsafetensors (NAO vem na imagem base).
#
# Build:
#   docker build -f docker/vllm-swap.Dockerfile -t vllm-swap . --no-cache
#   # reprodutivel no commit atual do fork (main):
#   docker build -f docker/vllm-swap.Dockerfile \
#     --build-arg LLAMA_SWAP_REF=3972f033518b025c97513e08b28ffb5210e48fe3 -t vllm-swap .
#   # ou aponte para uma tag/release do fork quando existir:
#   docker build -f docker/vllm-swap.Dockerfile \
#     --build-arg LLAMA_SWAP_REF=v0.1-gpu -t vllm-swap .
#
# O llama-swap e COMPILADO a partir do fork eduardofbrito/llama-swap,
# porque as features dele nao estao em nenhuma release upstream
# (commits 8f2b0b2..3972f03, todos em main):
#   - seletor de GPU por modelo + pagina GPUs na UI
#   - pagina GPUs mostra memoria usada/total por device e marca
#     "Used externally" a GPU ocupada por processo alheio ao llama-swap
#   - aba Conf: edicao do config.yaml pela UI (ver --enable-config-api abaixo)
#   - manualOnly: modelo que nunca carrega sob demanda (503 rapido)
#   - recentPoolSize: pool LRU, carregar um modelo nao derruba todos os outros
#   - vramCheck: recusa load quando a GPU nao tem memoria livre suficiente
#   - grupos com `gpus`: distribui os membros entre GPUs, um por device
# Build em 3 estagios:
#   1. node:24-slim   -> build da UI (Svelte/Vite)
#   2. golang:1.27.1  -> go build -tags embed_ui (UI embutida no binario)
#   3. vllm base      -> fastsafetensors/numba + binario final
#
# NOTA DE SINTAXE: nada de heredoc nem de string Python multi-linha aqui.
# O builder classico encerra o RUN em qualquer linha que nao termine em "\",
# e passa a interpretar o conteudo do script como instrucao Dockerfile.
# Por isso toda chamada Python abaixo cabe em uma linha so.
# =============================================================================

# v0.29.0 e a ultima release estavel (CUDA 13.0, como o v0.28.0). O guard
# do passo 3 confirma em tempo de build que o suporte a MTP/speculative
# decoding para modelos gated-delta-net (Qwen3.8) esta disponivel.
#
#   A) Estavel (default) -> --speculative-config pode ser habilitado se o
#      guard do build reportar "MTP suportado"
#   B) Nightly (se necessario):
#        docker pull vllm/vllm-openai:cu129-nightly
#        docker inspect vllm/vllm-openai:cu129-nightly \
#          --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
#        docker build --build-arg VLLM_IMAGE=vllm/vllm-openai:cu129-nightly-<sha> \
#                     --build-arg MTP_VLLM=0.27.2 ...
ARG VLLM_IMAGE=vllm/vllm-openai:v0.29.0

# Qwen3.8-Flash-Next (arquitetura Qwen4Exp) so tem suporte de verdade nesta
# imagem de preview — nao em nenhuma release numerada. Confirmado na pratica:
# a v0.29.0 ja RECONHECE a arquitetura (PR #53896 mesclou), mas o offload da
# tabela N-gram/PLE pra RAM do host (PR #53899) NAO mesclou em lugar nenhum
# oficial ainda — sem ele, os ~95 GiB da tabela tentam ir pra GPU e estouram
# em qualquer H100 unica. So esta imagem, com o offload como patch de preview,
# resolve isso. Ver stage "flash-next-vllm" abaixo — o binario/venv dela e
# copiado ISOLADO pra dentro da imagem final, sem misturar com o vLLM
# principal (versoes de FlashInfer e CUDA-ponto divergem entre as duas).
ARG FLASH_NEXT_VLLM_IMAGE=vllm/vllm-openai:qwen38-flash-next

# ---------------------------------------------------------------------------
# Estagio 1: build da UI (Svelte 5 / Vite)
# ---------------------------------------------------------------------------
FROM node:24-slim AS ui

ARG LLAMA_SWAP_REF=3972f033518b025c97513e08b28ffb5210e48fe3

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates git; \
    rm -rf /var/lib/apt/lists/*; \
    git clone --filter=blob:none https://github.com/eduardofbrito/llama-swap.git /src/llama-swap; \
    git -C /src/llama-swap checkout --quiet ${LLAMA_SWAP_REF}

WORKDIR /src/llama-swap/ui
RUN set -eux; \
    npm ci --no-audit --no-fund; \
    npm run build

# ---------------------------------------------------------------------------
# Estagio 2: compila o binario com a UI embutida
# ---------------------------------------------------------------------------
FROM golang:1.27.1 AS go-build

ARG LLAMA_SWAP_REF=3972f033518b025c97513e08b28ffb5210e48fe3

WORKDIR /src/llama-swap
RUN set -eux; \
    git clone --filter=blob:none https://github.com/eduardofbrito/llama-swap.git .; \
    git checkout --quiet ${LLAMA_SWAP_REF}
# UI do estagio 1 -> pasta que o build Go embute (go:embed all:ui_dist)
COPY --from=ui /src/llama-swap/internal/server/ui_dist ./internal/server/ui_dist
RUN set -eux; \
    GOOS=linux GOARCH=amd64 CGO_ENABLED=0 \
    go build -tags embed_ui \
      -ldflags="-X main.commit=${LLAMA_SWAP_REF} -X main.version=$(git describe --abbrev=6 --tags 2>/dev/null || git rev-parse --short HEAD) -X main.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      -o /out/llama-swap .

# ---------------------------------------------------------------------------
# Estagio 2b: filesystem da imagem de preview do Flash-Next, so pra COPY
#
# Nao roda nada aqui — e so uma referencia pro estagio final copiar o
# venv/dist-packages dela isolado. Puxar essa imagem so pra isso e caro
# (~15GB+), mas e o preco de ter as duas versoes de vLLM convivendo numa
# imagem so, sem Docker-outside-of-Docker.
# ---------------------------------------------------------------------------
FROM ${FLASH_NEXT_VLLM_IMAGE} AS flash-next-vllm

# ---------------------------------------------------------------------------
# Estagio 3: imagem final sobre o vLLM
# ---------------------------------------------------------------------------
FROM ${VLLM_IMAGE}

ARG MIN_VLLM=0.29.0
ARG MIN_TRANSFORMERS=5.10.4
ARG MTP_VLLM=0.27.2
# Reparo do numba: auto (tenta consertar) | skip (nao mexe)
ARG FIX_NUMBA=auto

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# -----------------------------------------------------------------------------
# 1) Dependencias de sistema
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates jq; \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2) fastsafetensors, sem perturbar o ambiente
#
# NAO usar `pip install --upgrade` cego. A imagem base tem um conjunto de
# versoes casadas (torch/numpy/numba/transformers); um upgrade pode subir o
# numpy e quebrar o numba. A v0.29.0 exige transformers>=5.10.4; a imagem
# base ja vem com uma versao >= disso, entao o piso e atendido sem instalar
# nada (o guard do passo 3 confere).
# -----------------------------------------------------------------------------
RUN set -eux; \
    if command -v uv >/dev/null 2>&1; then PIP="uv pip install --system --no-cache"; else PIP="pip install --no-cache-dir"; fi; \
    python3 -c "from importlib.metadata import version; print(chr(10).join(p+'=='+version(p) for p in ('numpy','torch','transformers')))" > /tmp/constraints.txt; \
    cat /tmp/constraints.txt; \
    $PIP -c /tmp/constraints.txt fastsafetensors; \
    rm -f /tmp/constraints.txt

# -----------------------------------------------------------------------------
# 2b) Reparo do numba (FIX_NUMBA=auto)
#
# A v0.29.0 pode vir de fabrica com numpy 2.5 + numba que so aceita <=2.4.
# Tentativas, da menos invasiva para a mais:
#   1. subir o numba (versao nova pode ja aceitar numpy 2.5)
#   2. baixar o numpy para <2.5
# O torch fica travado nas duas, para o pip nao arrastar meio ambiente junto.
# Se nada funcionar, segue em frente: numba ausente e degradacao, nao erro.
# -----------------------------------------------------------------------------
RUN set -eux; \
    if [ "$FIX_NUMBA" = "skip" ]; then echo "FIX_NUMBA=skip — nada a fazer"; exit 0; fi; \
    if python3 -c "import numba" 2>/dev/null; then echo "numba ja importa — nada a fazer"; exit 0; fi; \
    if command -v uv >/dev/null 2>&1; then PIP="uv pip install --system --no-cache"; else PIP="pip install --no-cache-dir"; fi; \
    python3 -c "from importlib.metadata import version; print('torch=='+version('torch'))" > /tmp/pin.txt; \
    echo "=== numba quebrado — tentativa 1: subir o numba ==="; \
    $PIP -c /tmp/pin.txt -U numba || true; \
    if python3 -c "import numba" 2>/dev/null; then echo "OK: corrigido subindo o numba"; rm -f /tmp/pin.txt; exit 0; fi; \
    echo "=== tentativa 2: baixar o numpy para <2.5 ==="; \
    $PIP -c /tmp/pin.txt "numpy<2.5" || true; \
    rm -f /tmp/pin.txt; \
    if python3 -c "import numba" 2>/dev/null; then echo "OK: corrigido baixando o numpy"; else echo "AVISO: numba segue quebrado — vLLM roda com caminhos mais lentos" >&2; fi

# -----------------------------------------------------------------------------
# 3) Verificacao — falha o build cedo, nao em producao as 3h
#    numba e MTP apenas avisam; os demais bloqueiam.
# -----------------------------------------------------------------------------
RUN set -eux; \
    python3 -c "import sys; from importlib.metadata import version as v; from packaging.version import Version as V; a=v('vllm'); print('vllm:', a); sys.exit('FALHA: vllm '+a+' < $MIN_VLLM') if V(a)<V('$MIN_VLLM') else None"; \
    python3 -c "import sys; from importlib.metadata import version as v; from packaging.version import Version as V; a=v('transformers'); print('transformers:', a); sys.exit('FALHA: transformers '+a+' < $MIN_TRANSFORMERS') if V(a)<V('$MIN_TRANSFORMERS') else None"; \
    python3 -c "import fastsafetensors; print('fastsafetensors: OK')"; \
    python3 -c "from importlib.metadata import version as v; import numba; print('numba:', v('numba'), 'OK (numpy', v('numpy')+')')" || echo "AVISO: numba nao importa — vLLM funciona, alguns caminhos ficam mais lentos" >&2; \
    python3 -c "from importlib.metadata import version as v; from packaging.version import Version as V; print('AVISO: MTP indisponivel — mantenha --speculative-config COMENTADO' if V(v('vllm'))<V('$MTP_VLLM') else 'OK: MTP suportado — pode habilitar --speculative-config')"; \
    echo "=== requisitos verificados ==="

# -----------------------------------------------------------------------------
# 3b) vLLM experimental do Flash-Next — ISOLADO, sem misturar com o principal
#
# So dist-packages, nao o /usr/local inteiro: torch e flashinfer modernos
# empacotam suas proprias libs CUDA dentro do wheel (nvidia-cublas-cuXX,
# nvidia-cudnn-cuXX etc. ja vao junto em dist-packages), entao normalmente
# nao dependem de uma instalacao de CUDA do sistema fora dali. O driver em
# si vem do host via nvidia-container-toolkit de qualquer forma, no runtime.
#
# O wrapper roda com o mesmo interprete /usr/bin/python3 da imagem base
# (v0.29.0), so trocando o PYTHONPATH pra resolver "import vllm" pro
# dist-packages isolado em vez do do sistema. Funciona porque ambas as
# imagens usam Python 3.12 (mesma ABI cp312) — NAO testado neste build; a
# checagem abaixo falha o build se o import nao resolver pra versao certa,
# mas nao substitui rodar de verdade contra uma GPU antes de confiar nisso
# em producao.
# -----------------------------------------------------------------------------
COPY --from=flash-next-vllm /usr/local/lib/python3.12/dist-packages /opt/vllm-flash-next/dist-packages

RUN set -eux; \
    printf '%s\n' \
      '#!/bin/bash' \
      '# Roda o vLLM experimental (build de preview com suporte a Qwen4Exp/PLE' \
      '# offload) isolado do vLLM principal da imagem. Uso: igual ao "vllm serve",' \
      '# so trocando o nome do comando.' \
      '#   vllm-flash-next serve aixiaoma/Qwen3.8-Flash-Next-W4A16 --port ${PORT} ...' \
      'set -euo pipefail' \
      'export PYTHONPATH="/opt/vllm-flash-next/dist-packages${PYTHONPATH:+:$PYTHONPATH}"' \
      'export PYTHONNOUSERSITE=1' \
      'exec python3 -m vllm.entrypoints.cli.main "$@"' \
      > /usr/local/bin/vllm-flash-next; \
    chmod +x /usr/local/bin/vllm-flash-next

# Checagem estatica, sem GPU: confirma que "import vllm" sob o wrapper
# resolve pro dist-packages isolado (nao pro vLLM 0.29.0 do sistema) e que
# o env var do offload da PLE existe NESSA copia especifica — e exatamente
# o diferencial que fizemos essa ginastica toda pra ter. Se isso falhar, o
# build para aqui em vez de descobrir em produção, 90 segundos antes do
# OOM de sempre.
RUN set -eux; \
    FLASH_VLLM_VER=$(PYTHONPATH=/opt/vllm-flash-next/dist-packages PYTHONNOUSERSITE=1 python3 -c "import vllm; print(vllm.__version__)"); \
    echo "vllm (flash-next, isolado): $FLASH_VLLM_VER"; \
    case "$FLASH_VLLM_VER" in \
      0.29.0|0.28.0) echo "FALHA: import resolveu pro vLLM do sistema ($FLASH_VLLM_VER), nao pro isolado — PYTHONPATH nao esta shadowing" >&2; exit 1 ;; \
    esac; \
    grep -q "VLLM_PLE_CPU_OFFLOAD" /opt/vllm-flash-next/dist-packages/vllm/envs.py \
      || { echo "FALHA: VLLM_PLE_CPU_OFFLOAD nao existe nessa copia do vLLM — a imagem apontada em FLASH_NEXT_VLLM_IMAGE pode nao ser a de preview esperada" >&2; exit 1; }; \
    echo "=== vllm-flash-next verificado (offload da PLE presente) ==="

# NAO RESOLVIDO POR ESTE BUILD: mesmo com o binario isolado disponivel, o
# config.yaml ainda precisa ser ajustado pra chamar "vllm-flash-next serve"
# em vez de "vllm serve" na entrada do modelo qwen3.8-flash-next — o
# wrapper so fica pronto pra ser usado, ele nao muda nada no config.yaml
# sozinho.

# -----------------------------------------------------------------------------
# 4) llama-swap (fork, compilado nos estagios 1+2 — UI embutida)
# -----------------------------------------------------------------------------
COPY --from=go-build /out/llama-swap /usr/local/bin/llama-swap
RUN set -eux; \
    chmod +x /usr/local/bin/llama-swap; \
    llama-swap --version

# -----------------------------------------------------------------------------
# 5) Caches persistentes — monte /cache como volume.
#    E o que faz o boot cair de ~4min30 para ~1min50 (DeepGEMM + torch.compile).
# -----------------------------------------------------------------------------
ENV HF_HOME=/cache/huggingface \
    VLLM_CACHE_ROOT=/cache/vllm \
    TRITON_CACHE_DIR=/cache/triton \
    OUTLINES_CACHE_DIR=/cache/outlines
RUN mkdir -p /cache/huggingface /cache/vllm /cache/triton /cache/outlines

# SEM `VOLUME ["/cache"]` de proposito. A instrucao VOLUME cria um volume
# ANONIMO novo a cada `docker run` quando nada e montado ali — ou seja, o
# cache nasce vazio toda vez e voce paga download de 28GB + torch.compile +
# DeepGEMM warmup em cada container novo, sem nenhum aviso.
#
# Monte SEMPRE explicitamente, com volume nomeado ou bind mount:
#   docker run -v vllm-swap-cache:/cache ...
#   docker run -v /u01/cache:/cache ...
#
# No compose:
#   volumes:
#     - vllm-swap-cache:/cache    # nomeado, sobrevive a recriacao
#   ou
#     - /u01/cache:/cache         # bind, voce ve os arquivos no host

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# A imagem base define ENTRYPOINT ["vllm", "serve"] — sobrescrevemos.
#
# /app/config/ vem do mount (compose) como um DIRETORIO:
#   - /opt/vllm-swap/config:/app/config
# O config fica em /app/config/config.yaml. Prefira bind de DIRETORIO: assim o
# config.yaml nao e um ponto de mount e o salvamento da UI faz tmp+rename
# atomico. Com bind de ARQUIVO o rename por cima do mount point falha com
# EBUSY; desde 339b50c existe fallback de escrita in-place (truncate+write no
# mesmo inode), entao funciona, mas sem a atomicidade do rename. Ler a aba Conf
# nunca escreve no arquivo, em nenhum dos dois casos.
#
# --enable-config-api liga os endpoints /api/config (aba Conf + dialogo Add
# Model). ELES SO FUNCIONAM COM apiKeys CONFIGURADO no config.yaml:
#
#   apiKeys:
#     - sk-sua-chave-aqui
#
# Sem apiKeys os endpoints respondem 403 e a UI esconde os dois controles — a
# flag sozinha NAO expoe superficie de escrita sem autenticacao. O motivo do
# gate duplo: quem consegue escrever um bloco de modelo escolhe o `cmd` dele e
# depois pode inicia-lo, o que e execucao de comando arbitrario no host.
# Remova a flag se voce nao quer editar o config pela UI nesta imagem.
#
# O seletor de GPU da UI usa GET /api/gpus e o query param llama-swap-gpu; sem
# GPU no host o seletor se oculta e o config.yaml continua mandando. Ele nao
# depende de --enable-config-api.
#
# A pagina GPUs mostra memoria usada/total por device, lida do monitor de
# performance. Se voce desligar performance no config.yaml, a pagina passa a
# dizer "No memory reading" e a marcacao "Used externally" some junto — e ela
# e justamente o jeito de ver uma GPU segurada pelo vllm-flash-next isolado ou
# por outro processo fora do llama-swap.
#
# As features de GPU/scheduler sao todas opt-in pelo config.yaml e NAO mudam
# nada por default. Para esta imagem (varias GPUs, vLLM pesado) as tres que
# valem a pena avaliar sao:
#
#   routing:
#     scheduler:
#       settings:
#         fifo:
#           # mantem os N modelos mais recentes carregados em vez de derrubar
#           # todos; N precisa caber de verdade na VRAM
#           recentPoolSize: 2
#           # recusa o load com 503 quando a GPU nao tem memoria livre — util
#           # aqui porque o vllm-flash-next isolado e outros processos podem
#           # segurar VRAM que o llama-swap nao conhece
#           vramCheck: true
#     router:
#       settings:
#         groups:
#           pool:
#             # distribui os membros entre as GPUs, um por device
#             gpus: ["0", "1"]
#             members: [modelA, modelB, modelC]
#
# Com vramCheck ligado, declare models.<id>.vramMB nos modelos grandes. Sem
# numero declarado NEM medido o modelo e sempre admitido (o guard nunca recusa
# por ignorancia), entao ate o primeiro load bem-sucedido a checagem nao faz
# nada para aquele modelo. O valor medido sai do ultimo load e e gravado no
# sqlite; sem store.path o banco e em memoria e a medicao morre com o
# processo. Para a medicao sobreviver, aponte store.path para dentro de
# /cache (que ja e volume):
#
#   store:
#     path: /cache/llama-swap.db
ENTRYPOINT ["llama-swap", \
            "--config", "/app/config/config.yaml", \
            "--listen", "0.0.0.0:8000", \
            "--enable-config-api"]
