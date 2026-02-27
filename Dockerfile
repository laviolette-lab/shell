ARG PY_VERSION=3.12

FROM python:${PY_VERSION}-slim AS base

# create non-root user (primarily for devcontainer)
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 -m vscode

# System libraries required by openslide, pyvips, and macenko-pca (OpenBLAS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenslide0 \
    libvips42 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install dependencies first for better layer caching
COPY pyproject.toml README.md LICENSE.txt ./
COPY src/ src/

RUN chown -R vscode:vscode /app

FROM base AS hatch
RUN pip3 install --no-cache-dir hatch hatch-uv
ENV HATCH_ENV=default
ENTRYPOINT ["hatch", "run"]

FROM base AS dev
COPY requirements/ requirements/
COPY requirements.txt ./
COPY tests/ tests/
COPY docs/ docs/
COPY mkdocs.yml ./
RUN pip3 install --no-cache-dir hatch hatch-uv \
    && hatch build \
    && pip3 install --no-cache-dir $(find /app -name 'requirement*.txt' -exec echo -n '-r {} ' \;)
USER vscode

FROM base AS prod
COPY --from=dev /app/dist/*.whl /tmp/
RUN pip3 install --no-cache-dir /tmp/*.whl \
    && rm -rf /tmp/*.whl
USER vscode
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import shell" || exit 1
