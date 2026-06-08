1. **Update `config/settings.py`**:
   - Use `replace_with_git_merge_diff` to add `cors_origins` and `allowed_hosts` to `Settings`, along with a `@field_validator(mode="before")` for parsing comma-separated lists. We will add the properties:
   ```python
    cors_origins: list[str] = Field(
        default_factory=lambda: ["*"], validation_alias="CORS_ORIGINS"
    )
    allowed_hosts: list[str] = Field(
        default_factory=lambda: ["*"], validation_alias="ALLOWED_HOSTS"
    )
   ```
   And the validator:
   ```python
    @field_validator("cors_origins", "allowed_hosts", mode="before")
    @classmethod
    def parse_comma_separated_list(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        if isinstance(v, list):
            return v
        if v is None:
            return ["*"]
        raise ValueError("Must be a comma-separated string or a list of strings")
   ```

2. **Verify `config/settings.py` update**:
   - Use `read_file` to verify the exact changes applied correctly.

3. **Update `api/app.py`**:
   - Use `replace_with_git_merge_diff` to import `CORSMiddleware` and `TrustedHostMiddleware`:
   ```python
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
   ```
   - Use `replace_with_git_merge_diff` to add the middlewares to the FastAPI app initialization. Ensure `TrustedHostMiddleware` is added last (outermost).
   ```python
    # Add CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials="*" not in settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Trusted Host Middleware (added last to execute first)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts,
    )
   ```

4. **Verify `api/app.py` update**:
   - Use `read_file` to check that the middlewares were added correctly and in the correct order.
   - Run `uv run ruff check` to ensure there are no linting errors.

5. **Run all relevant tests (e.g., `uv run pytest`)**
   - Ensure the tests pass after the security enhancement.

6. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
