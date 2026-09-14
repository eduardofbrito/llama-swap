# =============================================================================
# vllm-swap — llama-swap (este fork) sobre vLLM
#
# O config.yaml usa --load-format fastsafetensors, que exige o pacote
# fastsafetensors (NAO vem na imagem base).
#
# Alem do vLLM, esta imagem carrega os demais runtimes que o llama-swap sabe
# orquestrar. Cada um e opcional via --build-arg WITH_<X>=0 (todos ligados por
# default). O que entra e quanto custa, aproximadamente, alem da base:
#
#   WITH_LLAMACPP=1    llama-server / llama-cli / llama-bench   ~0,3 GB
#   WITH_WHISPERCPP=1  whisper-server (ASR)                     ~0,1 GB
#   WITH_AUDIOCPP=1    audiocpp_server (TTS/ASR ggml)           ~0,2 GB
#   (os tres acima compartilham ~0,7 GB de libs CUDA em /opt/ggml/lib)
#   WITH_OLLAMA=1      binario oficial + runners CUDA proprios  ~2,5 GB
#   WITH_COMFYUI=1     venv proprio com torch                   ~8 GB
#   WITH_KOKORO=1      venv proprio com torch + modelo baked    ~7 GB
#   WITH_QWEN3TTS=1    venv proprio com torch                   ~7 GB
#
# Com tudo ligado a imagem passa de 60 GB. Para uma variante enxuta:
#   docker build --build-arg WITH_COMFYUI=0 --build-arg WITH_OLLAMA=0 \
#                --build-arg WITH_KOKORO=0  --build-arg WITH_QWEN3TTS=0 \
#                -f docker/vllm-swap.Dockerfile -t vllm-swap-slim .
#
# Build:
#   docker build -f docker/vllm-swap.Dockerfile -t vllm-swap . --no-cache
#   # reprodutivel no commit atual do fork (main, checado em 2026-09-14):
#   docker build -f docker/vllm-swap.Dockerfile \
#     --build-arg LLAMA_SWAP_REF=cfbce3cfe02c3a90240b0217b16e34963f54d59c -t vllm-swap .
#   # ou aponte para uma tag/release do fork quando existir:
#   docker build -f docker/vllm-swap.Dockerfile \
#     --build-arg LLAMA_SWAP_REF=v0.1-gpu -t vllm-swap .
#
# O llama-swap e COMPILADO a partir do fork eduardofbrito/llama-swap,
# porque as features dele nao estao em nenhuma release upstream
# (commits 8f2b0b2..cfbce3c, todos em main; HEAD checado em 2026-09-14):
#   - seletor de GPU por modelo + pagina GPUs na UI
#   - aba Conf: edicao do config.yaml pela UI (ver --enable-config-api abaixo)
#   - manualOnly: modelo que nunca carrega sob demanda (503 rapido)
#   - recentPoolSize: pool LRU, carregar um modelo nao derruba todos os outros
#   - vramCheck: recusa load quando a GPU nao tem memoria livre suficiente
#   - grupos com `gpus`: distribui os membros entre GPUs, um por device
#   - sleepMode: despejo poe o modelo pra dormir em vez de matar — o processo
#     fica vivo com os pesos na RAM do host e a proxima request ACORDA ele em
#     segundos. E a feature que mais muda a vida nesta imagem, porque um start
#     de vLLM grande custa minutos (spawn + pesos + torch.compile + CUDA graphs
#     + warmup) e o wake pula tudo menos a copia. Ver o bloco de exemplo no fim
#     deste arquivo: precisa de VLLM_SERVER_DEV_MODE=1 e --enable-sleep-mode
#   - uiApiKeys: chave separada pro dashboard/API de controle, distinta da
#     chave de inferencia (ver --enable-config-api abaixo)
#   (3 commits novos desde o pin anterior, 56bea6b: separacao apiKeys/uiApiKeys,
#   correcao dos caminhos do Kokoro no wrapper, e o sleepMode)
# Build em estagios:
#   1. node:24-slim   -> build da UI (Svelte/Vite)
#   2. golang:1.27.1  -> go build -tags embed_ui (UI embutida no binario)
#   2b. imagem preview -> so filesystem, pro vLLM experimental do Flash-Next
#   2c. cuda devel    -> llama.cpp, whisper.cpp e audio.cpp (ggml, CUDA)
#   2d. debian slim   -> tarball oficial do Ollama
#   2e/f/g. vllm base -> venvs isolados de ComfyUI, Kokoro e Qwen3-TTS
#   3. vllm base      -> fastsafetensors/numba + binario final + tudo acima
#
# NOTA DE SINTAXE: nada de heredoc nem de string Python multi-linha aqui.
# O builder classico encerra o RUN em qualquer linha que nao termine em "\",
# e passa a interpretar o conteudo do script como instrucao Dockerfile.
# Por isso toda chamada Python abaixo cabe em uma linha so. Os estagios novos
# seguem a mesma regra: scripts sao escritos com printf '%s\n', nunca heredoc.
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
# Runtimes adicionais — todos opcionais, todos ligados por default.
#
# CUDA_DEVEL_IMAGE precisa casar com a imagem final em DOIS eixos:
#   - CUDA maior (13.x aqui, como a base v0.29.0) — senao o libcudart copiado
#     nao e o que os binarios pedem
#   - versao do Ubuntu / glibc — um binario compilado no 24.04 nao roda numa
#     base 22.04. O guard do passo 8 roda `ldd` em cada binario, e um
#     descasamento de glibc aparece ali como "not found", falhando o build em
#     vez de virar erro de runtime. Atencao: o ldd prova que as bibliotecas
#     resolvem, nao que o binario executa — um build que caiu pra CPU em
#     silencio passa (so o audio.cpp tem checagem do link de CUDA).
#
# CMAKE_CUDA_ARCHITECTURES=90 e Hopper (H100), a arquitetura de destino desta
# imagem.
# Compilar so a arquitetura da casa deixa o build muito mais rapido e a imagem
# menor. Se a imagem for rodar em outro host, adicione o numero dele aqui
# (ex.: "89;90" pra Ada + Hopper) — arquiteturas ausentes ainda rodam por JIT
# do PTX mais proximo, pagando o custo na primeira execucao.
# ---------------------------------------------------------------------------
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG CMAKE_CUDA_ARCHITECTURES=90

# NCCL: o ggml (tanto o do llama.cpp quanto o vendorizado no audio.cpp) tem
# GGML_CUDA_NCCL ligado por DEFAULT, e a imagem CUDA devel traz libnccl-dev —
# entao os binarios linkam NCCL sem ninguem pedir, e a lib (centenas de MB)
# passa a ter que viajar junto pra imagem final.
#
# Aqui isso fica DESLIGADO porque o proveito seria nulo neste host: o
# placement por grupo do fork coloca um modelo por GPU, e NCCL so serve para
# um mesmo processo falar com varias GPUs. O proprio llama.cpp trata a
# ausencia como aviso, nao erro ("performance for multiple CUDA GPUs will be
# suboptimal"). Ligue com --build-arg GGML_CUDA_NCCL=ON se algum dia rodar um
# GGUF grande dividido entre placas.
ARG GGML_CUDA_NCCL=OFF

ARG WITH_LLAMACPP=1
ARG WITH_WHISPERCPP=1
ARG WITH_AUDIOCPP=1
ARG WITH_OLLAMA=1
ARG WITH_COMFYUI=1
ARG WITH_KOKORO=1
ARG WITH_QWEN3TTS=1

# Pins. Nenhum destes projetos promete estabilidade entre commits; a data ao
# lado e quando o pin foi conferido. Suba um de cada vez.
ARG LLAMACPP_REF=b10941
ARG WHISPERCPP_REF=v1.9.4
ARG AUDIOCPP_REF=v0.7.4
ARG COMFYUI_REF=master
ARG OLLAMA_VERSION=v0.34.0
ARG KOKORO_REF=master
ARG QWEN3TTS_REF=main

# Indice de wheels do torch para os venvs isolados. Vazio = PyPI, que hoje
# entrega o torch com runtime CUDA 12.8 embutido no proprio wheel — roda em
# qualquer driver recente, inclusive num host com CUDA 13, e nao depende de
# download.pytorch.org estar liberado na rede. Aponte para
# https://download.pytorch.org/whl/cu130 se quiser casar o ponto do CUDA.
ARG TORCH_INDEX_URL=

# ---------------------------------------------------------------------------
# Estagio 1: build da UI (Svelte 5 / Vite)
# ---------------------------------------------------------------------------
FROM node:24-slim AS ui

ARG LLAMA_SWAP_REF=cfbce3cfe02c3a90240b0217b16e34963f54d59c

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

ARG LLAMA_SWAP_REF=cfbce3cfe02c3a90240b0217b16e34963f54d59c

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
# Estagio 2c: llama.cpp, whisper.cpp e audio.cpp (familia ggml, CUDA)
#
# Os tres saem do mesmo estagio porque compartilham toolchain e, no fim, o
# mesmo conjunto de libs CUDA — copia-las uma vez em /opt/ggml/lib e o que
# evita triplicar ~700 MB de cuBLAS na imagem final.
#
# Cada componente e um `if` dentro do RUN, nao um estagio proprio: com
# WITH_X=0 o compile e pulado, mas a imagem CUDA devel ainda e baixada. Se
# voce nao quer NENHUM dos tres e se incomoda com esse download, comente este
# estagio inteiro e o COPY correspondente no passo 6.
#
# whisper.cpp e compilado SEM ffmpeg de proposito (WHISPER_FFMPEG=OFF): ligar
# isso faz o binario linkar libavcodec/libavformat da imagem de build, cujas
# sonames nao necessariamente batem com as da imagem final. A imagem final tem
# o executavel ffmpeg instalado, entao converta a entrada pra WAV 16 kHz antes
# de mandar pro whisper-server quando ela nao for WAV.
# ---------------------------------------------------------------------------
FROM ${CUDA_DEVEL_IMAGE} AS ggml-build

ARG CMAKE_CUDA_ARCHITECTURES
ARG GGML_CUDA_NCCL
ARG WITH_LLAMACPP
ARG WITH_WHISPERCPP
ARG WITH_AUDIOCPP
ARG LLAMACPP_REF
ARG WHISPERCPP_REF
ARG AUDIOCPP_REF

# bash + pipefail: os guards abaixo usam pipe (ldd|awk|grep, readelf|grep). Com
# o /bin/sh default uma falha no meio do pipe fica mascarada e o guard passa.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# /out existe sempre, mesmo com os tres desligados — o COPY do estagio final
# nao pode depender de um ARG.
RUN set -eux; \
    mkdir -p /out/bin /out/lib /out/share; \
    echo "=== espaco livre neste estagio de build ==="; df -h / /tmp || true; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get -o Acquire::Retries=3 update || { echo "FALHA no apt-get update. Se a mensagem foi \"At least one invalid signature was encountered\" em TODOS os repositorios (inclusive archive.ubuntu.com, nao so o da NVIDIA), o problema quase nunca e chave de GPG: e disco cheio no host do Docker. O InRelease chega truncado e a assinatura nao confere. Confira 'df -h /var/lib/docker' e libere espaco ('docker system prune -af --volumes'). Este build precisa de ~150 GB livres com tudo ligado." >&2; exit 1; }; \
    apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates pkg-config libgomp1 binutils; \
    rm -rf /var/lib/apt/lists/*

# --- llama.cpp -------------------------------------------------------------
RUN set -eux; \
    if [ "$WITH_LLAMACPP" != "1" ]; then echo "WITH_LLAMACPP=0 — pulando llama.cpp"; exit 0; fi; \
    git clone --filter=blob:none https://github.com/ggml-org/llama.cpp.git /src/llama.cpp; \
    git -C /src/llama.cpp checkout --quiet ${LLAMACPP_REF}; \
    cmake -S /src/llama.cpp -B /src/llama.cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF \
      -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_CUDA_NCCL="${GGML_CUDA_NCCL}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -lcuda"; \
    cmake --build /src/llama.cpp/build --config Release -j"$(nproc)" --target llama-server llama-cli llama-bench; \
    for b in llama-server llama-cli llama-bench; do test -f "/src/llama.cpp/build/bin/$b" || { echo "FALHA: $b nao foi gerado" >&2; exit 1; }; cp "/src/llama.cpp/build/bin/$b" /out/bin/; done; \
    rm -rf /src/llama.cpp

# --- whisper.cpp -----------------------------------------------------------
RUN set -eux; \
    if [ "$WITH_WHISPERCPP" != "1" ]; then echo "WITH_WHISPERCPP=0 — pulando whisper.cpp"; exit 0; fi; \
    git clone --filter=blob:none https://github.com/ggml-org/whisper.cpp.git /src/whisper.cpp; \
    git -C /src/whisper.cpp checkout --quiet ${WHISPERCPP_REF}; \
    cmake -S /src/whisper.cpp -B /src/whisper.cpp/build \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DWHISPER_FFMPEG=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DGGML_CUDA=ON -DGGML_VULKAN=OFF -DGGML_CUDA_NCCL="${GGML_CUDA_NCCL}" \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -lcuda" \
      "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -lcuda"; \
    cmake --build /src/whisper.cpp/build --config Release -j"$(nproc)" --target whisper-server whisper-cli; \
    for b in whisper-server whisper-cli; do test -f "/src/whisper.cpp/build/bin/$b" || { echo "FALHA: $b nao foi gerado" >&2; exit 1; }; cp "/src/whisper.cpp/build/bin/$b" /out/bin/; done; \
    find /src/whisper.cpp/build \( -name "*.so" -o -name "*.so.*" \) -exec cp -a {} /out/lib/ \; ; \
    rm -rf /src/whisper.cpp

# --- audio.cpp -------------------------------------------------------------
# AUDIOCPP_DEPLOYMENT_BUILD=ON compila o catalogo model_specs/ dentro do
# binario; sem isso um pacote safetensors falha com "model spec not found".
# O build fica estatico de proposito, pra nao jogar um terceiro libggml*.so
# ABI-incompativel em cima dos do whisper.cpp.
RUN set -eux; \
    if [ "$WITH_AUDIOCPP" != "1" ]; then echo "WITH_AUDIOCPP=0 — pulando audio.cpp"; exit 0; fi; \
    git clone --filter=blob:none https://github.com/0xShug0/audio.cpp.git /src/audio.cpp; \
    git -C /src/audio.cpp checkout --quiet ${AUDIOCPP_REF}; \
    cmake -S /src/audio.cpp -B /src/audio.cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_MODEL_SET=full \
      -DENGINE_ENABLE_NATIVE_CPU=OFF -DENGINE_ENABLE_OPENMP=ON \
      -DENGINE_BUILD_EXAMPLES=OFF -DENGINE_BUILD_TESTS=OFF -DENGINE_BUILD_WARMBENCH=OFF \
      -DENGINE_ENABLE_CUDA=ON -DENGINE_ENABLE_CUDA_GRAPHS=ON -DENGINE_ENABLE_VULKAN=OFF \
      -DGGML_CUDA_NCCL="${GGML_CUDA_NCCL}" \
      -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES}" \
      -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler \
      "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -lcuda"; \
    cmake --build /src/audio.cpp/build --config Release -j"$(nproc)" --target audiocpp_server audiocpp_cli; \
    for b in audiocpp_server audiocpp_cli; do test -f "/src/audio.cpp/build/bin/$b" || { echo "FALHA: $b nao foi gerado" >&2; exit 1; }; readelf -d "/src/audio.cpp/build/bin/$b" | grep NEEDED | grep -qE 'libcudart\.so\.1[23]' || { echo "FALHA: $b nao linkou runtime CUDA 12/13 — o build caiu pra CPU em silencio" >&2; exit 1; }; cp "/src/audio.cpp/build/bin/$b" /out/bin/; done; \
    mkdir -p /out/share/audiocpp; \
    cp -r /src/audio.cpp/model_specs /out/share/audiocpp/model_specs; \
    rm -rf /src/audio.cpp

# --- libs CUDA que os binarios acima precisam em runtime ---------------------
# Coletadas por ldd, nao por lista fixa: se um upgrade passar a pedir libcufft
# ou libnvrtc, ela vem junto sem ninguem lembrar de editar aqui.
#
# A busca NAO se limita a /usr/local/cuda. Foi assim que a primeira versao
# deste arquivo deixou libnccl.so.2 pra tras: o pacote libnccl2 instala em
# /usr/lib/x86_64-linux-gnu, fora da arvore do CUDA, e o filtro por caminho
# nao a via. Agora o criterio e o NOME da biblioteca (libcu*, libnccl*,
# libnv*, libcudnn*), venha ela de onde vier.
#
# libcuda.so.1 (o driver) e os stubs ficam de fora DE PROPOSITO — quem entrega
# o driver e o host, via nvidia-container-toolkit. Copiar o stub daqui produz
# um container que sobe e nao ve GPU nenhuma.
#
# A lista de nomes e explicita, e nao um prefixo curto tipo "libcu*": esse
# prefixo tambem casa com libcurl, e mandar a libcurl da imagem de build pra
# imagem final sombreia a da base com uma ABI que pode nao bater.
#
# ldconfig -n no fim recria os links de SONAME (libfoo.so.1 -> libfoo.so.1.2.3)
# dentro de /out/lib. Sem isso um binario que pede o SONAME nao acha a lib,
# mesmo com o arquivo real ali do lado.
RUN set -eux; \
    if [ -z "$(ls -A /out/bin 2>/dev/null)" ]; then echo "nenhum binario ggml — nada de lib pra copiar"; exit 0; fi; \
    for b in /out/bin/*; do LD_LIBRARY_PATH=/out/lib ldd "$b" 2>/dev/null || true; done \
      | awk '/=> \// {print $3}' \
      | grep -vE '/stubs/|/libcuda\.so' \
      | grep -E '/(libcudart|libcublas|libcudnn|libcufft|libcurand|libcusparse|libcusolver|libcupti|libcufile|libnccl|libnvrtc|libnvjitlink|libnvToolsExt|libnvperf)[^/]*$' \
      | sort -u > /tmp/cudalibs.txt; \
    echo "=== libs NVIDIA que viajam com os binarios ==="; cat /tmp/cudalibs.txt; \
    while read -r l; do [ -n "$l" ] && cp -L "$l" /out/lib/; done < /tmp/cudalibs.txt; \
    cp -L /usr/lib/x86_64-linux-gnu/libgomp.so.1 /out/lib/ 2>/dev/null || true; \
    ldconfig -n /out/lib; \
    echo "=== /out/lib ==="; ls -la /out/lib; du -sh /out/lib

# Guard do proprio estagio: repete a checagem que a imagem final faz, mas aqui,
# onde o erro aponta pra causa. Uma lib que o filtro acima nao pegou falha o
# build no estagio que a produziu, em vez de 40 minutos depois.
RUN set -eux; \
    fail=0; \
    for b in /out/bin/*; do \
      [ -f "$b" ] || continue; \
      missing=$(LD_LIBRARY_PATH=/out/lib ldd "$b" 2>/dev/null | grep 'not found' | grep -v 'libcuda\.so' || true); \
      if [ -n "$missing" ]; then echo "FALHA: $(basename "$b") ficou sem:" >&2; echo "$missing" >&2; fail=1; fi; \
    done; \
    [ "$fail" = "0" ] || { echo "Alguma lib nao foi coletada. Se o nome comeca com lib{cu,nccl,nv,cudnn} o filtro do RUN anterior precisa ser ampliado; se e uma lib do proprio projeto (libwhisper, libggml), o build dele voltou a gerar shared libs e o -DBUILD_SHARED_LIBS=OFF nao pegou." >&2; exit 1; }; \
    echo "=== estagio ggml: binarios com todas as dependencias resolvidas ==="

# ---------------------------------------------------------------------------
# Estagio 2d: Ollama (binario oficial)
#
# O tarball oficial ja traz os runners CUDA do Ollama em lib/ollama/. Ele NAO
# usa o vLLM nem o llama.cpp desta imagem — e um servidor inteiro, com daemon
# proprio e catalogo de modelos proprio. Por isso ele entra como arvore
# isolada em /opt/ollama e nao em /usr.
#
# O asset mudou de nome: hoje e .tar.zst (era .tgz). Baixamos do GitHub, e nao
# de ollama.com/download, para pinar a versao de verdade.
# ---------------------------------------------------------------------------
FROM debian:trixie-slim AS ollama-dl

ARG WITH_OLLAMA
ARG OLLAMA_VERSION

RUN set -eux; \
    mkdir -p /out/ollama; \
    if [ "$WITH_OLLAMA" != "1" ]; then echo "WITH_OLLAMA=0 — pulando Ollama"; exit 0; fi; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends curl ca-certificates zstd tar; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL -o /tmp/ollama.tar.zst "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"; \
    tar --zstd -xf /tmp/ollama.tar.zst -C /out/ollama; \
    rm -f /tmp/ollama.tar.zst; \
    test -x /out/ollama/bin/ollama || { echo "FALHA: bin/ollama nao veio no tarball — o layout do asset mudou" >&2; ls -R /out/ollama >&2; exit 1; }; \
    ls -la /out/ollama/bin

# ---------------------------------------------------------------------------
# Estagios 2e/2f/2g: servicos Python, cada um no seu venv
#
# Os tres partem da PROPRIA imagem final (${VLLM_IMAGE}) por dois motivos:
#   1. o interprete e o mesmo, entao o venv criado aqui funciona la — um venv
#      feito com outro python3.12 aponta pra um binario que nao existe na
#      imagem final e quebra na primeira execucao
#   2. a imagem ja foi baixada, entao o estagio nao custa download nenhum
#
# Sao venvs SEM --system-site-packages: cada servico traz o proprio torch.
# Isso e caro em disco (~7 GB cada) e e o ponto: ComfyUI quer
# transformers>=4.50, o Qwen3-TTS pina transformers==4.57.3 e o vLLM 0.29
# exige >=5.10.4. Qualquer tentativa de compartilhar site-packages termina
# com um `pip` bem-intencionado derrubando o vLLM em producao.
# ---------------------------------------------------------------------------
FROM ${VLLM_IMAGE} AS comfyui-build

ARG WITH_COMFYUI
ARG COMFYUI_REF
ARG TORCH_INDEX_URL

RUN set -eux; \
    mkdir -p /out/comfyui; \
    if [ "$WITH_COMFYUI" != "1" ]; then echo "WITH_COMFYUI=0 — pulando ComfyUI"; exit 0; fi; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends git ca-certificates python3-venv; \
    rm -rf /var/lib/apt/lists/*; \
    git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI.git /out/comfyui/app; \
    git -C /out/comfyui/app checkout --quiet ${COMFYUI_REF}; \
    python3 -m venv /out/comfyui/venv; \
    /out/comfyui/venv/bin/pip install --no-cache-dir --upgrade pip wheel; \
    if [ -n "$TORCH_INDEX_URL" ]; then /out/comfyui/venv/bin/pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio; fi; \
    /out/comfyui/venv/bin/pip install --no-cache-dir -r /out/comfyui/app/requirements.txt; \
    /out/comfyui/venv/bin/python -c "import torch, comfy; print('comfyui deps OK, torch', torch.__version__)" 2>/dev/null || /out/comfyui/venv/bin/python -c "import torch; print('torch', torch.__version__)"; \
    find /out/comfyui/venv -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

FROM ${VLLM_IMAGE} AS kokoro-build

ARG WITH_KOKORO
ARG KOKORO_REF
# O dicionario UniDic (japones) sao ~526 MB. Desligado por default porque a
# casa serve pt-BR e ingles; ligue se precisar de ja.
ARG KOKORO_JAPANESE=0
# Modelo baked na imagem (~330 MB) pra o container nao depender de rede no
# primeiro boot. Com 0 o entrypoint do Kokoro baixa no primeiro start.
ARG KOKORO_DOWNLOAD_MODEL=1

RUN set -eux; \
    mkdir -p /out/kokoro; \
    if [ "$WITH_KOKORO" != "1" ]; then echo "WITH_KOKORO=0 — pulando Kokoro"; exit 0; fi; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends git ca-certificates python3-venv espeak-ng espeak-ng-data libsndfile1; \
    rm -rf /var/lib/apt/lists/*; \
    git clone --filter=blob:none https://github.com/remsky/Kokoro-FastAPI.git /out/kokoro/app; \
    git -C /out/kokoro/app checkout --quiet ${KOKORO_REF}; \
    command -v uv >/dev/null 2>&1 || python3 -m pip install --no-cache-dir --break-system-packages uv || python3 -m pip install --no-cache-dir uv; \
    PY=$(command -v python3.12 || command -v python3); echo "interprete da base: $PY"; \
    cd /out/kokoro/app && UV_PROJECT_ENVIRONMENT=/out/kokoro/venv UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 uv sync --frozen --extra gpu --no-install-project --python "$PY" --python-preference only-system; \
    test -x /out/kokoro/venv/bin/python || { echo "FALHA: venv do Kokoro nao foi criado onde esperado" >&2; exit 1; }; \
    if [ "$KOKORO_DOWNLOAD_MODEL" = "1" ]; then cd /out/kokoro/app && /out/kokoro/venv/bin/python docker/scripts/download_model.py --output api/src/models/v1_0; fi; \
    if [ "$KOKORO_JAPANESE" = "1" ]; then /out/kokoro/venv/bin/python -m unidic download; fi; \
    find /out/kokoro -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

FROM ${VLLM_IMAGE} AS qwen3tts-build

ARG WITH_QWEN3TTS
ARG QWEN3TTS_REF
ARG TORCH_INDEX_URL

# O fork groxaxo/Qwen3-TTS-Openai-Fastapi pina transformers==4.57.3 no
# pyproject — e exatamente por isso que ele nao pode dividir site-packages com
# o vLLM. flash-attn NAO e instalado: compila em ~20 min e o backend
# "official" roda sem ele.
RUN set -eux; \
    mkdir -p /out/qwen3-tts; \
    if [ "$WITH_QWEN3TTS" != "1" ]; then echo "WITH_QWEN3TTS=0 — pulando Qwen3-TTS"; exit 0; fi; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends git ca-certificates python3-venv ffmpeg libsndfile1 sox libsox-dev build-essential; \
    rm -rf /var/lib/apt/lists/*; \
    git clone --filter=blob:none https://github.com/groxaxo/Qwen3-TTS-Openai-Fastapi.git /out/qwen3-tts/app; \
    git -C /out/qwen3-tts/app checkout --quiet ${QWEN3TTS_REF}; \
    python3 -m venv /out/qwen3-tts/venv; \
    /out/qwen3-tts/venv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools; \
    if [ -n "$TORCH_INDEX_URL" ]; then /out/qwen3-tts/venv/bin/pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" torch torchaudio; fi; \
    /out/qwen3-tts/venv/bin/pip install --no-cache-dir "/out/qwen3-tts/app[api]"; \
    /out/qwen3-tts/venv/bin/python -c "import torch, transformers; print('qwen3-tts deps OK — torch', torch.__version__, 'transformers', transformers.__version__)"; \
    find /out/qwen3-tts -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

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
#
# Alem do basico do vLLM, aqui entram as libs de runtime dos servicos de audio
# e imagem: espeak-ng (fonemizador do Kokoro), libsndfile/sox/ffmpeg (leitura e
# escrita de audio no Kokoro e no Qwen3-TTS), libgomp (OpenMP do audio.cpp).
# Sao pacotes apt — nenhum deles toca no ambiente Python pinado do vLLM.
#
# O link de espeak-ng-data e o mesmo truque do Dockerfile oficial do Kokoro: o
# phonemizer procura os dados em /usr/share/espeak-ng-data, o Debian instala em
# /usr/lib/<arch>/espeak-ng-data.
# -----------------------------------------------------------------------------
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates jq \
      ffmpeg libsndfile1 sox espeak-ng espeak-ng-data libgomp1; \
    rm -rf /var/lib/apt/lists/*; \
    mkdir -p /usr/share/espeak-ng-data; \
    ln -sfn /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ 2>/dev/null || true

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
#    numba, MTP e sleep mode apenas avisam; os demais bloqueiam.
#
# O sleep mode so avisa porque e opt-in por modelo: sem ele a imagem funciona
# igual, o llama-swap recebe 404 no /sleep e para o modelo como sempre fez —
# perde-se a troca rapida, nada mais. Mas e melhor descobrir aqui do que
# depois de configurar sleepMode: true e nao entender por que o swap continua
# levando minutos.
# -----------------------------------------------------------------------------
RUN set -eux; \
    python3 -c "import sys; from importlib.metadata import version as v; from packaging.version import Version as V; a=v('vllm'); print('vllm:', a); sys.exit('FALHA: vllm '+a+' < $MIN_VLLM') if V(a)<V('$MIN_VLLM') else None"; \
    python3 -c "import sys; from importlib.metadata import version as v; from packaging.version import Version as V; a=v('transformers'); print('transformers:', a); sys.exit('FALHA: transformers '+a+' < $MIN_TRANSFORMERS') if V(a)<V('$MIN_TRANSFORMERS') else None"; \
    python3 -c "import fastsafetensors; print('fastsafetensors: OK')"; \
    python3 -c "from importlib.metadata import version as v; import numba; print('numba:', v('numba'), 'OK (numpy', v('numpy')+')')" || echo "AVISO: numba nao importa — vLLM funciona, alguns caminhos ficam mais lentos" >&2; \
    python3 -c "from importlib.metadata import version as v; from packaging.version import Version as V; print('AVISO: MTP indisponivel — mantenha --speculative-config COMENTADO' if V(v('vllm'))<V('$MTP_VLLM') else 'OK: MTP suportado — pode habilitar --speculative-config')"; \
    python3 -c "import dataclasses, vllm.envs as e; from vllm.engine.arg_utils import EngineArgs; flag='enable_sleep_mode' in {f.name for f in dataclasses.fields(EngineArgs)}; dev=hasattr(e, 'VLLM_SERVER_DEV_MODE'); print('OK: sleep mode disponivel — models.*.sleepMode funciona com VLLM_SERVER_DEV_MODE=1 + --enable-sleep-mode' if (flag and dev) else 'AVISO: sleep mode indisponivel neste vLLM (--enable-sleep-mode=%s, VLLM_SERVER_DEV_MODE=%s) — sleepMode: true vai cair no fallback de parar o modelo' % (flag, dev))" || echo "AVISO: nao foi possivel verificar o sleep mode neste vLLM" >&2; \
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
    OUTLINES_CACHE_DIR=/cache/outlines \
    NUMBA_CACHE_DIR=/cache/numba \
    OLLAMA_MODELS=/cache/ollama
RUN mkdir -p /cache/huggingface /cache/vllm /cache/triton /cache/outlines /cache/numba /cache/ollama

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

# =============================================================================
# 6) Runtimes adicionais
#
# Cada arvore vem de um estagio proprio. Com WITH_<X>=0 o estagio produz um
# diretorio vazio e o COPY nao traz nada — os wrappers do passo 7 tambem nao
# sao criados, entao o comando simplesmente nao existe na imagem.
#
# Nada aqui e ativado por default em RUNTIME: o llama-swap so executa o que
# estiver escrito no config.yaml. Um binario presente e inerte.
# =============================================================================
COPY --from=ggml-build    /out            /opt/ggml
COPY --from=ollama-dl     /out/ollama     /opt/ollama
COPY --from=comfyui-build /out/comfyui    /opt/comfyui
COPY --from=kokoro-build  /out/kokoro     /opt/kokoro
COPY --from=qwen3tts-build /out/qwen3-tts /opt/qwen3-tts

# -----------------------------------------------------------------------------
# 7) Wrappers em /usr/local/bin
#
# Os binarios ggml precisam de LD_LIBRARY_PATH=/opt/ggml/lib (as libs CUDA
# viajaram junto com eles, e nao estao no ld.so.conf da imagem). Os servicos
# Python precisam entrar no venv certo. Em vez de exigir isso do config.yaml,
# cada um ganha um wrapper de uma linha — assim o `cmd` do modelo fica igual
# ao que a documentacao de cada projeto mostra.
# -----------------------------------------------------------------------------
ARG WITH_LLAMACPP
ARG WITH_WHISPERCPP
ARG WITH_AUDIOCPP
ARG WITH_OLLAMA
ARG WITH_COMFYUI
ARG WITH_KOKORO
ARG WITH_QWEN3TTS

# --- ggml (llama.cpp / whisper.cpp / audio.cpp) ------------------------------
RUN set -eux; \
    for b in /opt/ggml/bin/*; do \
      [ -f "$b" ] || continue; \
      n=$(basename "$b"); \
      chmod +x "$b"; \
      printf '%s\n' \
        '#!/bin/sh' \
        '# wrapper gerado no build: poe as libs CUDA que viajaram com o binario' \
        '# no caminho antes de exec. Ver passo 7 do Dockerfile.' \
        'export LD_LIBRARY_PATH="/opt/ggml/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' \
        "exec /opt/ggml/bin/$n \"\$@\"" \
        > "/usr/local/bin/$n"; \
      chmod +x "/usr/local/bin/$n"; \
    done; \
    n=0; for w in /opt/ggml/bin/*; do [ -f "$w" ] && { echo "wrapper: $(basename "$w")"; n=$((n+1)); }; done; [ "$n" -gt 0 ] || echo "nenhum binario ggml nesta imagem"

# --- Ollama ------------------------------------------------------------------
# O tarball oficial ja traz os runners em lib/ollama, resolvidos pelo binario
# em relacao ao proprio caminho — por isso bin/ e lib/ ficam lado a lado em
# /opt/ollama e nao sao espalhados por /usr.
#
# OLLAMA_MODELS aponta pra /cache (volume), senao o catalogo baixado morre com
# o container. OLLAMA_HOST NAO e fixado aqui: quem escolhe a porta e o
# llama-swap, via ${PORT} no config.yaml.
RUN set -eux; \
    if [ "$WITH_OLLAMA" != "1" ] || [ ! -x /opt/ollama/bin/ollama ]; then echo "sem Ollama nesta imagem"; exit 0; fi; \
    printf '%s\n' \
      '#!/bin/sh' \
      '# wrapper gerado no build. Uso tipico no config.yaml do llama-swap:' \
      '#   cmd: ollama serve      (com OLLAMA_HOST=127.0.0.1:${PORT} no env)' \
      'export OLLAMA_MODELS="${OLLAMA_MODELS:-/cache/ollama}"' \
      'exec /opt/ollama/bin/ollama "$@"' \
      > /usr/local/bin/ollama; \
    chmod +x /usr/local/bin/ollama

# --- ComfyUI -----------------------------------------------------------------
# --base-directory manda models/, custom_nodes/, input/, output/ e user/ pra
# fora da imagem. O default aponta pra /models/comfyui; sobrescreva passando
# --base-directory no cmd do modelo.
RUN set -eux; \
    if [ "$WITH_COMFYUI" != "1" ] || [ ! -x /opt/comfyui/venv/bin/python ]; then echo "sem ComfyUI nesta imagem"; exit 0; fi; \
    mkdir -p /models/comfyui; \
    printf '%s\n' \
      '#!/bin/sh' \
      '# wrapper gerado no build: entra no venv isolado do ComfyUI.' \
      '#   comfyui-server --listen 127.0.0.1 --port ${PORT} --cuda-device 0' \
      'set -eu' \
      'cd /opt/comfyui/app' \
      'case " $* " in *" --base-directory "*) ;; *) set -- "$@" --base-directory /models/comfyui ;; esac' \
      'exec /opt/comfyui/venv/bin/python main.py "$@"' \
      > /usr/local/bin/comfyui-server; \
    chmod +x /usr/local/bin/comfyui-server

# --- Kokoro (TTS) ------------------------------------------------------------
# Servidor OpenAI-compativel: /v1/audio/speech e /v1/audio/voices. Aceita
# --host/--port porque quem sobe e o uvicorn.
#
# MODEL_DIR/VOICES_DIR: o Kokoro-FastAPI tem esses caminhos HARDCODED como
# default em api/src/core/config.py ("/app/api/src/models",
# "/app/api/src/voices/v1_0") — pensados pra imagem oficial deles, onde o
# app mora em /app. Aqui o app fica em /opt/kokoro/app (isolado do resto),
# entao sem sobrescrever essas duas env vars o processo sobe, procura o
# modelo em /app/... (que nao existe nesta imagem) e morre com
# "Model files not found" mesmo com o .pth baixado no build. Pydantic-settings
# le as env vars direto, sem prefixo, casando por nome de forma
# case-insensitive — e o campo e device_type, nao device (DEVICE=gpu era
# ignorado em silencio, e o device ficava por conta do auto-detect).
RUN set -eux; \
    if [ "$WITH_KOKORO" != "1" ] || [ ! -x /opt/kokoro/venv/bin/python ]; then echo "sem Kokoro nesta imagem"; exit 0; fi; \
    printf '%s\n' \
      '#!/bin/sh' \
      '# wrapper gerado no build: sobe o Kokoro-FastAPI no venv isolado.' \
      '#   kokoro-server --host 127.0.0.1 --port ${PORT}' \
      'set -eu' \
      'cd /opt/kokoro/app' \
      'export PYTHONPATH="/opt/kokoro/app:/opt/kokoro/app/api${PYTHONPATH:+:$PYTHONPATH}"' \
      'export MODEL_DIR="${MODEL_DIR:-/opt/kokoro/app/api/src/models}"' \
      'export VOICES_DIR="${VOICES_DIR:-/opt/kokoro/app/api/src/voices/v1_0}"' \
      'export USE_GPU="${USE_GPU:-true}" DEVICE_TYPE="${DEVICE_TYPE:-cuda}"' \
      'export PHONEMIZER_ESPEAK_PATH="${PHONEMIZER_ESPEAK_PATH:-/usr/bin}"' \
      'export PHONEMIZER_ESPEAK_DATA="${PHONEMIZER_ESPEAK_DATA:-/usr/share/espeak-ng-data}"' \
      'export ESPEAK_DATA_PATH="${ESPEAK_DATA_PATH:-/usr/share/espeak-ng-data}"' \
      'exec /opt/kokoro/venv/bin/python -m uvicorn api.src.main:app "$@"' \
      > /usr/local/bin/kokoro-server; \
    chmod +x /usr/local/bin/kokoro-server

# --- Qwen3-TTS ---------------------------------------------------------------
# Este nao le --host/--port: ele so olha as variaveis HOST e PORT. O wrapper
# traduz as flags pra env, pra o cmd no config.yaml ficar igual ao dos outros.
RUN set -eux; \
    if [ "$WITH_QWEN3TTS" != "1" ] || [ ! -x /opt/qwen3-tts/venv/bin/python ]; then echo "sem Qwen3-TTS nesta imagem"; exit 0; fi; \
    printf '%s\n' \
      '#!/bin/sh' \
      '# wrapper gerado no build: sobe o Qwen3-TTS-Openai-Fastapi no venv' \
      '# isolado. O servidor le HOST/PORT do ambiente, entao traduzimos as' \
      '# flags aqui:  qwen3-tts-server --host 127.0.0.1 --port ${PORT}' \
      'set -eu' \
      'while [ $# -gt 0 ]; do case "$1" in --host) HOST="$2"; shift 2 ;; --port) PORT="$2"; shift 2 ;; --backend) TTS_BACKEND="$2"; shift 2 ;; *) echo "qwen3-tts-server: argumento nao reconhecido: $1" >&2; exit 2 ;; esac; done' \
      'cd /opt/qwen3-tts/app' \
      'export HOST="${HOST:-127.0.0.1}" PORT="${PORT:-8880}"' \
      'export TTS_BACKEND="${TTS_BACKEND:-official}"' \
      'export PYTHONPATH="/opt/qwen3-tts/app${PYTHONPATH:+:$PYTHONPATH}"' \
      'exec /opt/qwen3-tts/venv/bin/python -m api.main' \
      > /usr/local/bin/qwen3-tts-server; \
    chmod +x /usr/local/bin/qwen3-tts-server

# -----------------------------------------------------------------------------
# 8) Guards dos runtimes adicionais
#
# A checagem principal e o ldd: um binario ggml compilado contra outra glibc
# ou outro ponto do CUDA linka limpo no builder e so falha quando alguem tenta
# carregar um modelo. Aqui isso vira erro de build.
#
# libcuda.so.1 e ignorado de proposito — ele so aparece em runtime, entregue
# pelo host via nvidia-container-toolkit. Durante o build ele SEMPRE consta
# como "not found", e isso e o comportamento correto.
# -----------------------------------------------------------------------------
RUN set -eux; \
    fail=0; \
    for b in /opt/ggml/bin/*; do \
      [ -f "$b" ] || continue; \
      missing=$(LD_LIBRARY_PATH=/opt/ggml/lib ldd "$b" 2>/dev/null | grep 'not found' | grep -v 'libcuda\.so' || true); \
      if [ -n "$missing" ]; then echo "FALHA: $(basename "$b") tem dependencia nao resolvida:" >&2; echo "$missing" >&2; fail=1; fi; \
    done; \
    [ "$fail" = "0" ] || { echo "Tres causas possiveis, nesta ordem: (a) uma lib NVIDIA que o filtro de coleta do estagio ggml-build nao pegou — amplie o grep de nomes la; (b) uma lib do proprio projeto (libwhisper, libggml) porque o build voltou a gerar shared libs; (c) CUDA_DEVEL_IMAGE nao casa com a base em versao de CUDA ou de Ubuntu/glibc. O guard equivalente dentro do ggml-build deveria ter pego (a) e (b) antes daqui." >&2; exit 1; }; \
    for b in /opt/ggml/bin/*; do [ -f "$b" ] && echo "ggml OK: $(basename "$b")"; done; \
    echo "=== binarios ggml verificados ==="

RUN set -eux; \
    if [ -x /opt/ollama/bin/ollama ]; then LD_LIBRARY_PATH=/opt/ollama/lib/ollama /opt/ollama/bin/ollama --version 2>&1 | head -2 || echo "AVISO: 'ollama --version' saiu diferente de zero (normal sem daemon)"; fi; \
    if [ -x /opt/comfyui/venv/bin/python ]; then /opt/comfyui/venv/bin/python -c "import torch; print('comfyui venv: torch', torch.__version__)"; fi; \
    if [ -x /opt/kokoro/venv/bin/python ]; then /opt/kokoro/venv/bin/python -c "import torch, uvicorn; print('kokoro venv: torch', torch.__version__)"; fi; \
    if [ -x /opt/qwen3-tts/venv/bin/python ]; then /opt/qwen3-tts/venv/bin/python -c "import torch, transformers; print('qwen3-tts venv: transformers', transformers.__version__)"; fi; \
    python3 -c "from importlib.metadata import version as v; print('vLLM do sistema segue intacto: transformers', v('transformers'))"; \
    echo "=== runtimes adicionais verificados ==="

# Os modelos ficam fora da imagem. /models e o ponto de montagem esperado
# pelos exemplos abaixo; monte junto com /cache:
#   -v /u01/models:/models -v /u01/cache:/cache
RUN mkdir -p /models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

# =============================================================================
# ENTRADAS DE EXEMPLO NO config.yaml
#
# Nenhum runtime desta imagem sobe sozinho — o llama-swap so executa o `cmd`
# que estiver escrito no config. Os blocos abaixo sao o ponto de partida de
# cada um. ${PORT} e substituido pelo llama-swap; sempre use 127.0.0.1 como
# host, porque quem fala com a rede e o proxy, nao o backend.
#
# models:
#   # --- vLLM com sleep mode (a troca rapida) ---------------------------------
#   # Os tres pedacos abaixo tem que estar TODOS presentes:
#   #   env VLLM_SERVER_DEV_MODE=1  -> sem isso o vLLM nem monta /sleep e
#   #                                  /wake_up, e o llama-swap leva 404
#   #   --enable-sleep-mode         -> a flag que habilita o recurso no engine
#   #   sleepMode: true             -> diz ao llama-swap pra DORMIR o modelo ao
#   #                                  despeja-lo, em vez de mata-lo
#   # Faltando qualquer um, nada quebra: o /sleep falha, o llama-swap loga e
#   # para o modelo como sempre fez. Voce so perde a troca rapida.
#   #
#   # O que muda: despejado, o processo continua vivo com os pesos na RAM do
#   # host e devolve a VRAM. A proxima request ACORDA ele — uma copia de volta
#   # pra GPU — em vez de pagar spawn + pesos + torch.compile + CUDA graphs +
#   # warmup de novo. Num modelo grande sao segundos no lugar de minutos.
#   #
#   # Custo: a RAM do host segura ~o tamanho dos pesos enquanto ele dorme. Dois
#   # modelos de 30 GB dormindo sao 60 GB de RAM. O `ttl` abaixo e o que limita
#   # isso: despejo dorme, mas TTL e unload explicito param o processo de vez.
#   qwen3-27b-fp8:
#     cmd: |
#       vllm serve /models/Qwen3-27B-FP8 --host 127.0.0.1 --port ${PORT}
#       --served-model-name qwen3-27b-fp8
#       --load-format fastsafetensors
#       --enable-sleep-mode
#     env: ["CUDA_VISIBLE_DEVICES=0", "VLLM_SERVER_DEV_MODE=1"]
#     sleepMode: true
#     ttl: 3600
#     vramMB: 30000
#
#   # ATENCAO de seguranca: VLLM_SERVER_DEV_MODE=1 monta, alem do sleep,
#   # endpoints administrativos (/collective_rpc, /reset_prefix_cache) na porta
#   # desse modelo. O llama-swap nao os expoe por conta propria, MAS o
#   # passthrough /upstream/<model>/... alcanca qualquer caminho do backend —
#   # entao quem tiver uma chave de inferencia valida alcanca esses endpoints
#   # tambem. Por isso a env var fica POR MODELO aqui, e nao um ENV global da
#   # imagem: so os modelos que realmente usam sleep mode ganham essa
#   # superficie. Se a instancia for exposta alem da sua rede, considere
#   # tambem separar as chaves com uiApiKeys (ver perto do ENTRYPOINT).
#
#   # --- llama.cpp -----------------------------------------------------------
#   qwen3-30b-gguf:
#     cmd: |
#       llama-server --host 127.0.0.1 --port ${PORT}
#       --model /models/Qwen3-30B-A3B-Q4_K_M.gguf
#       --n-gpu-layers 99 --ctx-size 32768 --jinja
#     env: ["CUDA_VISIBLE_DEVICES=0"]
#
#   # --- Ollama --------------------------------------------------------------
#   # O Ollama e um daemon com catalogo proprio, nao um processo por modelo.
#   # O padrao aqui e uma entrada que sobe o daemon; o modelo em si voce escolhe
#   # pelo nome na requisicao, e o Ollama carrega sob demanda.
#   ollama:
#     cmd: ollama serve
#     env: ["OLLAMA_HOST=127.0.0.1:${PORT}", "CUDA_VISIBLE_DEVICES=1"]
#     checkEndpoint: /api/tags
#     # Combine com manualOnly + um grupo persistent se quiser ele sempre de pe.
#
#   # --- whisper.cpp (ASR) ---------------------------------------------------
#   # Compilado sem ffmpeg: mande WAV 16 kHz mono. Para outro formato, converta
#   # antes (o executavel ffmpeg esta na imagem).
#   whisper-large-v3:
#     cmd: |
#       whisper-server --host 127.0.0.1 --port ${PORT}
#       --model /models/ggml-large-v3-turbo.bin
#     env: ["CUDA_VISIBLE_DEVICES=2"]
#
#   # --- audio.cpp (TTS/ASR ggml) --------------------------------------------
#   # Precisa de um server.json descrevendo os modelos; veja
#   # docker/unified/audiocpp-server.example.json no repositorio do llama-swap.
#   audiocpp:
#     cmd: |
#       audiocpp_server --host 127.0.0.1 --port ${PORT}
#       --config /models/audiocpp-server.json --backend cuda --no-ui
#     env: ["CUDA_VISIBLE_DEVICES=2"]
#
#   # --- ComfyUI -------------------------------------------------------------
#   # O llama-swap tem endpoint dedicado /comfyui/ para este caso.
#   comfyui:
#     cmd: comfyui-server --listen 127.0.0.1 --port ${PORT} --cuda-device 0
#     checkEndpoint: /system_stats
#     ttl: 600
#
#   # --- Kokoro (TTS) --------------------------------------------------------
#   kokoro:
#     cmd: kokoro-server --host 127.0.0.1 --port ${PORT}
#     env: ["CUDA_VISIBLE_DEVICES=3"]
#     checkEndpoint: /health
#
#   # --- Qwen3-TTS -----------------------------------------------------------
#   qwen3-tts:
#     cmd: qwen3-tts-server --host 127.0.0.1 --port ${PORT}
#     env: ["CUDA_VISIBLE_DEVICES=3"]
#     checkEndpoint: /health
#
# Os dois TTS respondem em /v1/audio/speech, entao o LiteLLM pode apontar pra
# ca em vez de pras stacks separadas — a diferenca e que sob o llama-swap eles
# passam a disputar (e liberar) GPU junto com os demais modelos, em vez de
# segurar VRAM o tempo todo.
# =============================================================================

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
# Model). ELES SO FUNCIONAM COM CHAVE CONFIGURADA no config.yaml — uiApiKeys,
# ou apiKeys como fallback:
#
#   # chave de inferencia: e o que LiteLLM e afins usam
#   apiKeys:
#     - sk-inferencia-aqui
#   # chave do dashboard e do /api/* (inclui a edicao de config). Quando
#   # presente, a chave de apiKeys acima NAO abre mais a UI — que e o ponto:
#   # o cliente externo nao consegue editar config nem descarregar modelos
#   uiApiKeys:
#     - sk-operador-aqui
#
# Sem nenhuma das duas os endpoints respondem 403 e a UI esconde os dois
# controles — a flag sozinha NAO expoe superficie de escrita sem autenticacao.
# O motivo do gate duplo: quem consegue escrever um bloco de modelo escolhe o
# `cmd` dele e depois pode inicia-lo, o que e execucao de comando arbitrario no
# host. Com esta imagem esse poder cresceu: o `cmd` agora alcanca llama.cpp,
# Ollama, ComfyUI e os dois TTS, nao so o vLLM. Remova a flag se voce nao quer
# editar o config pela UI nesta imagem.
#
# Se voce so definir apiKeys, o comportamento e o de antes: uma chave unica
# para tudo. Separar passa a valer a pena quando alguma coisa externa tem a
# chave de inferencia — ainda mais nesta imagem, onde modelos com
# VLLM_SERVER_DEV_MODE=1 expoem endpoints administrativos via /upstream.
#
# O seletor de GPU da UI usa GET /api/gpus e o query param llama-swap-gpu; sem
# GPU no host o seletor se oculta e o config.yaml continua mandando. Ele nao
# depende de --enable-config-api.
#
# As features de GPU/scheduler sao todas opt-in pelo config.yaml e NAO mudam
# nada por default. Para esta imagem (varias GPUs, vLLM pesado) as que valem a
# pena avaliar sao:
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
# ... mais o models.<id>.sleepMode, que e por modelo (ver o exemplo de vLLM
# acima). Os tres formam uma hierarquia de memoria e e assim que faz sentido
# pensar neles nesta imagem, onde um start de vLLM custa minutos:
#
#   VRAM   <- recentPoolSize: os modelos mais quentes nem saem da placa
#   RAM    <- sleepMode: os empurrados pra fora do pool dormem, wake em segundos
#   morto  <- ttl: depois de ocioso o bastante, devolve a RAM tambem
#
# Vale dizer o que NAO dorme: unload explicito (botao da UI, /unload) e
# expiracao de ttl param o processo de verdade. So o despejo — dar lugar a
# outro modelo — e que dorme. Um modelo dormindo aparece como "sleeping" na
# pagina Models e para de contar na GPU dele (ele nao segura VRAM nenhuma),
# entao uma placa cujo unico modelo dorme aparece como idle.
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
#
# Um aviso especifico do Ollama: ele NAO respeita o vramCheck nem o
# recentPoolSize para os modelos que ele proprio carrega. Do ponto de vista do
# llama-swap ha um processo so ("ollama serve"); o que acontece dentro dele e
# invisivel. Se o Ollama entrar em producao aqui, prenda-o a uma GPU propria
# via CUDA_VISIBLE_DEVICES — assim a pagina GPUs o mostra como "Used
# externally" no device dele em vez de contaminar a contabilidade dos outros.
ENTRYPOINT ["llama-swap", \
            "--config", "/app/config/config.yaml", \
            "--listen", "0.0.0.0:8000", \
            "--enable-config-api"]
