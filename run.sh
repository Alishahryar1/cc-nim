#!/bin/zsh
# Deprecated: Use 'brew services start cc-proxy' or 'cc-nim start' instead.
# This script remains for backward compatibility but may be removed in a future version.
uv run uvicorn server:app --host 0.0.0.0 --port 8082
