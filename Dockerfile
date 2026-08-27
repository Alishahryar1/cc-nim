# vim: ft=Dockerfile

FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN uv sync --frozen

ENTRYPOINT ["uv", "run", "fcc-server"]

# build:
# docker build -t fcc-image .

# run:
# docker run -d --rm -p 8082:8082 --env-file .env fcc-image:latest
# docker run -d --rm -p 8082:8082 --env LOCAL_ONLY_ADMIN=false fcc-image:latest