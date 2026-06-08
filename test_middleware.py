import os

from config.settings import Settings

os.environ["CORS_ORIGINS"] = "http://localhost,https://example.com"
os.environ["ALLOWED_HOSTS"] = "localhost, example.com"
s = Settings()
print(s.cors_origins)
print(s.allowed_hosts)
