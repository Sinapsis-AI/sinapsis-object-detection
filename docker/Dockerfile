FROM sinapsis-nvidia:base

RUN apt-get update && apt-get install -y git && apt-get clean

COPY . /app/
WORKDIR /app
RUN uv sync --frozen
RUN uv pip install sinapsis-object-detection[all] --extra-index-url https://pypi.sinapsis.tech
