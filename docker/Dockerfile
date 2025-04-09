FROM sinapsis-nvidia:base

COPY . /app/
WORKDIR /app
RUN uv sync --frozen
RUN uv pip install sinapsis-object-detection[all] --extra-index-url https://pypi.sinapsis.tech
