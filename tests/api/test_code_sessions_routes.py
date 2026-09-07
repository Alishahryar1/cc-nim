import asyncio
import uuid

import httpx
import pytest
import pytest_asyncio

from free_claude_code.application.code_sessions import CodeService
from free_claude_code.runtime.code_sessions_sqlite import SQLiteCodeStore
from tests.api.support import create_test_app
from tests.code_sessions_support import FakeHarness


@pytest_asyncio.fixture
async def code_api(tmp_path):
    harness = FakeHarness()
    code = CodeService(
        SQLiteCodeStore(tmp_path / "code.db", tmp_path / "code.lock"), harness
    )
    await code.start()
    app = create_test_app(code=code)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
        ) as client:
            yield client, code, harness, tmp_path, app
    finally:
        await code.close()
        await app.state.services.admin.close()


async def create_session(code_api):
    client, _, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "codex"},
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.asyncio
async def test_code_library_has_one_harness_and_no_native_work_on_open(code_api):
    client, _, harness, _, _ = code_api
    bootstrap = await client.get("/admin/api/code/bootstrap")
    assert bootstrap.json()["harnesses"] == [{"id": "codex", "name": "Codex"}]
    session = await create_session(code_api)
    detail = await client.get(f"/admin/api/code/sessions/{session['id']}")
    assert detail.status_code == 200
    assert detail.headers["cache-control"] == "no-store"
    assert detail.json()["items"] == []
    assert harness.connections == []
    page = await client.get(f"/admin/code/{session['id']}")
    assert page.status_code == 200


@pytest.mark.asyncio
async def test_http_send_competition_and_safe_item_projection(code_api):
    client, code, harness, _, _ = code_api
    session = await create_session(code_api)
    path = f"/admin/api/code/sessions/{session['id']}"
    responses = await asyncio.gather(
        *(
            client.post(
                path + "/turns",
                json={
                    "operation_id": str(uuid.uuid4()),
                    "expected_revision": session["revision"],
                    "expected_epoch": code.epoch,
                    "text": text,
                },
            )
            for text in ("one", "two")
        )
    )
    assert sorted(response.status_code for response in responses) == [202, 409]
    await harness.started.wait()
    await harness.connections[0].text(
        "turn-1", "answer", "**safe** <script>bad()</script>", complete=True
    )
    await harness.connections[0].finish("turn-1")
    detail = (await client.get(path)).json()
    item = detail["items"][-1]
    assert "<strong>safe</strong>" in item["html"]
    assert "<script>" not in item["html"]
    assert "raw" not in item
    assert detail["run"]["status"] == "completed"


@pytest.mark.asyncio
async def test_code_commands_reject_unplanned_fields_and_other_harnesses(code_api):
    client, code, _, folder, _ = code_api
    response = await client.post(
        "/admin/api/code/sessions",
        json={"session_id": str(uuid.uuid4()), "cwd": str(folder), "harness": "cline"},
    )
    assert response.status_code == 422
    session = await create_session(code_api)
    response = await client.post(
        f"/admin/api/code/sessions/{session['id']}/turns",
        json={
            "operation_id": str(uuid.uuid4()),
            "expected_revision": session["revision"],
            "expected_epoch": code.epoch,
            "text": "hi",
            "model": "override",
        },
    )
    assert response.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers", [{"host": "remote.example"}, {"origin": "https://remote.example"}]
)
async def test_code_routes_share_admin_access_boundary(code_api, headers):
    client, _, harness, _, _ = code_api
    for path in (
        "/admin/code",
        "/admin/api/code/bootstrap",
        "/admin/api/code/sessions",
        "/admin/api/code/events",
    ):
        response = await client.get(path, headers=headers)
        assert response.status_code == 403
    assert harness.connections == []
