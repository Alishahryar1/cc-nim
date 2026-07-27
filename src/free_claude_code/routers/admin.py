import threading

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/admin/api/credentials", tags=["admin"])
admin_lock = threading.Lock()

CREDENTIAL_STORAGE = {"claude": {"policy": "rotate", "keys": []}}


class KeyUpdateRequest(BaseModel):
    keys: list[str]
    policy: str | None = None


@router.put("/{env}/keys")
def update_credential_keys(env: str, payload: KeyUpdateRequest) -> dict:
    # Atomic Admin Key Updates using a lock to prevent concurrent tab overwrite race conditions
    with admin_lock:
        if env not in CREDENTIAL_STORAGE:
            CREDENTIAL_STORAGE[env] = {"policy": "rotate", "keys": []}

        CREDENTIAL_STORAGE[env]["keys"] = payload.keys
        if payload.policy:
            CREDENTIAL_STORAGE[env]["policy"] = payload.policy.lower()

        return {
            "status": "success",
            "env": env,
            "policy": CREDENTIAL_STORAGE[env]["policy"],
            "keys_count": len(CREDENTIAL_STORAGE[env]["keys"]),
        }
