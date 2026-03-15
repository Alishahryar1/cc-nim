FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:/root/.cargo/bin:$PATH"

# ── system deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git curl ca-certificates nodejs npm \
    && rm -rf /var/lib/apt/lists/*

# ── install uv ─────────────────────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ── install claude code cli (or stub if npm fails) ────────────────────────────
RUN npm install -g @anthropic-ai/claude-code --ignore-scripts 2>/dev/null || \
    ( mkdir -p /usr/local/bin && \
      printf '#!/usr/bin/env node\nprocess.exit(0);\n' > /usr/local/bin/claude && \
      chmod +x /usr/local/bin/claude )

# ── copy repo ──────────────────────────────────────────────────────────────────
COPY . /repo
WORKDIR /repo

# ── run install script ─────────────────────────────────────────────────────────
ENV HOME=/root
RUN FREE_CLAUDE_INSTALL_DIR=/opt/free-claude-code bash install.sh

# ── test entrypoint ────────────────────────────────────────────────────────────
COPY test/docker-entrypoint.sh /test.sh
RUN chmod +x /test.sh
CMD ["/test.sh"]
