"""hfsandbox provider — the local auth-injecting proxy (HTTP, SSE, WebSocket)."""
from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers.hfsandbox import (
    HfSandboxProvider,
    _AuthProxy,
    _build_provider_config,
    _Live,
    parse_volume_spec,
)

HEADERS = {"Authorization": "Bearer hf_test", "X-Sandbox-Token": "sbx_test"}


async def _upstream_app():
    seen: list[dict] = []

    async def status(request):
        seen.append(dict(request.headers))
        return web.json_response({"ok": True, "path": request.path, "q": request.query.get("q")})

    async def cmd(request):
        body = await request.json()
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(f"data: {json.dumps({'echo': body})}\n\n".encode())
        await resp.write_eof()
        return resp

    async def ws(request):
        seen.append(dict(request.headers))
        sock = web.WebSocketResponse()
        await sock.prepare(request)
        async for msg in sock:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await sock.send_str("echo:" + msg.data)
            elif msg.type == aiohttp.WSMsgType.BINARY:
                await sock.send_bytes(msg.data[::-1])
        return sock

    app = web.Application()
    app.router.add_get("/proxy/5000/status", status)
    app.router.add_post("/proxy/5000/cmd", cmd)
    app.router.add_get("/proxy/5000/ws", ws)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"127.0.0.1:{port}", seen


async def test_proxy_injects_headers_and_relays_http_sse_ws():
    up_runner, base, seen = await _upstream_app()
    proxy = _AuthProxy(f"https://{base}/proxy/5000", HEADERS)
    # the test upstream is plain http/ws, not TLS
    proxy._upstream = lambda path_qs, *, scheme: (  # type: ignore[method-assign]
        {"https://": "http://", "wss://": "ws://"}[scheme] + f"{base}/proxy/5000" + path_qs
    )
    await proxy.start()
    try:
        async with aiohttp.ClientSession() as cs:
            # plain GET with query string
            async with cs.get(f"{proxy.url}/status?q=1") as r:
                assert r.status == 200
                assert await r.json() == {"ok": True, "path": "/proxy/5000/status", "q": "1"}
            # streamed POST (SSE-style body)
            async with cs.post(f"{proxy.url}/cmd", json={"command": "echo hi"}) as r:
                assert r.status == 200
                text = await r.text()
                assert json.loads(text.split("data: ", 1)[1]) == {"echo": {"command": "echo hi"}}
            # websocket relay, both frame types
            async with cs.ws_connect(f"{proxy.url}/ws") as sock:
                await sock.send_str("ping")
                assert (await sock.receive()).data == "echo:ping"
                await sock.send_bytes(b"abc")
                assert (await sock.receive()).data == b"cba"
        # every upstream request carried the sandbox auth headers
        assert len(seen) == 2
        for h in seen:
            assert h["Authorization"] == HEADERS["Authorization"]
            assert h["X-Sandbox-Token"] == HEADERS["X-Sandbox-Token"]
    finally:
        await proxy.stop()
        await up_runner.cleanup()


def test_build_provider_config_defaults_and_overrides():
    cfg = _build_provider_config({})
    assert (cfg.image, cfg.flavor, cfg.idle_timeout) == ("ale-ubuntu22-docker", "cpu-upgrade", 3600)
    assert (cfg.transport, cfg.job_timeout) == ("job", "24h")
    assert cfg.resolution == (1024, 768)
    cfg = _build_provider_config({"flavor": "cpu-basic", "idle_timeout": "2h",
                                  "resolution": [800, 600], "namespace": "org"})
    assert (cfg.flavor, cfg.idle_timeout, cfg.resolution, cfg.namespace) == ("cpu-basic", "2h", (800, 600), "org")
    assert _build_provider_config({"transport": "sandbox"}).transport == "sandbox"
    assert _build_provider_config({"job_command": ["python", "-V"]}).job_command == ("python", "-V")
    with pytest.raises(ValueError, match="job_command"):
        _build_provider_config({"job_command": "python -V"})
    with pytest.raises(ValueError):
        _build_provider_config({"transport": "docker"})


def test_proxy_upstream_url_building():
    proxy = _AuthProxy("https://abc--5000.hf.jobs/", {})
    assert proxy._upstream("/cmd?x=1", scheme="https://") == "https://abc--5000.hf.jobs/cmd?x=1"
    assert proxy._upstream("/ws", scheme="wss://") == "wss://abc--5000.hf.jobs/ws"
    sub = _AuthProxy("https://h.hf.jobs/v1/proxy/5000", {})
    assert sub._upstream("/status", scheme="https://") == "https://h.hf.jobs/v1/proxy/5000/status"


def test_parse_volume_spec():
    v = parse_volume_spec("hf://buckets/ns/data:/mnt/data:ro")
    assert (v.type, v.source, v.mount_path, v.read_only, v.path) == ("bucket", "ns/data", "/mnt/data", True, None)
    v = parse_volume_spec("hf://buckets/ns/data/sub/dir:/mnt/x")
    assert (v.read_only, v.path) == (None, "sub/dir")
    v = parse_volume_spec("hf://datasets/org/ds:/ds")
    assert (v.type, v.read_only) == ("dataset", True)
    for bad in ("buckets/ns/x:/m", "hf://buckets/ns/x", "hf://buckets/ns/x:/m:rx", "hf://buckets/onlyns:/m"):
        with pytest.raises(ValueError):
            parse_volume_spec(bad)
    cfg = _build_provider_config({"volumes": ["hf://buckets/ns/data:/mnt/data:ro"]})
    assert cfg.volumes == ("hf://buckets/ns/data:/mnt/data:ro",)


async def test_job_is_cancelled_when_startup_poll_fails():
    class FakeApi:
        def __init__(self):
            self.cancelled = []
            self.command = None

        def run_job(self, **kwargs):
            self.command = kwargs["command"]
            return SimpleNamespace(id="job-1", owner=SimpleNamespace(name="owner"))

        def inspect_job(self, **kwargs):
            raise ConnectionError("temporary status failure")

        def cancel_job(self, **kwargs):
            self.cancelled.append(kwargs)

    api = FakeApi()
    image = SimpleNamespace(docker_entrypoint="/entry", cua_server_port=5000)
    with patch("huggingface_hub.HfApi", return_value=api), patch(
        "huggingface_hub.get_token", return_value="test-token"
    ), pytest.raises(ConnectionError, match="temporary status failure"):
        await HfSandboxProvider({"job_command": ["/bin/bash", "/entry"]})._create_via_job(
            "image", image, {}, SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo")
        )
    assert api.cancelled == [{"job_id": "job-1", "namespace": "owner"}]
    assert api.command == ["/bin/bash", "/entry"]


async def test_published_ale_image_uses_reachable_entrypoint():
    class FakeApi:
        def run_job(self, **kwargs):
            self.command = kwargs["command"]
            return SimpleNamespace(id="job-1", owner=SimpleNamespace(name="owner"))

        def inspect_job(self, **kwargs):
            raise ConnectionError("stop after command check")

        def cancel_job(self, **kwargs):
            pass

    api = FakeApi()
    image = SimpleNamespace(docker_entrypoint="/dockerstartup/entrypoint.sh", cua_server_port=5000)
    with patch("huggingface_hub.HfApi", return_value=api), patch(
        "huggingface_hub.get_token", return_value="test-token"
    ), pytest.raises(ConnectionError, match="stop after command check"):
        await HfSandboxProvider({})._create_via_job(
            "agentslastexam/ale-ubuntu22-docker:latest", image, {},
            SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo"),
        )
    assert api.command[:2] == ["/bin/bash", "-lc"]
    assert "--host 0.0.0.0 --port 5000" in api.command[2]


async def test_cancel_during_job_creation_cancels_late_job():
    started = threading.Event()
    finish_creation = threading.Event()
    cancelled = threading.Event()

    class FakeApi:
        def run_job(self, **kwargs):
            started.set()
            if not finish_creation.wait(5):
                raise TimeoutError("test job creation stalled")
            return SimpleNamespace(id="job-1", owner=SimpleNamespace(name="owner"))

        def cancel_job(self, **kwargs):
            assert kwargs == {"job_id": "job-1", "namespace": "owner"}
            cancelled.set()

    image = SimpleNamespace(docker_entrypoint="/entry", cua_server_port=5000)
    with patch("huggingface_hub.HfApi", return_value=FakeApi()), patch(
        "huggingface_hub.get_token", return_value="test-token"
    ):
        task = asyncio.create_task(HfSandboxProvider({})._create_via_job(
            "image", image, {}, SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo")
        ))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            finish_creation.set()
        assert await asyncio.to_thread(cancelled.wait, 5)


async def test_cua_wait_stops_when_job_has_failed():
    class Unreachable:
        id = "job-1"

        def __init__(self):
            self.metadata = {"transport": "job", "job_owner": "owner"}

        async def check_reachable(self, *, label):
            raise ConnectionError("port unavailable")

    class FakeApi:
        def inspect_job(self, **kwargs):
            assert kwargs == {"job_id": "job-1", "namespace": "owner"}
            return SimpleNamespace(status=SimpleNamespace(stage="ERROR", message="exec format error"))

    with patch("huggingface_hub.HfApi", return_value=FakeApi()), pytest.raises(
        RuntimeError, match="ended before CUA ready"
    ):
        await HfSandboxProvider({"cua_ready_timeout": 120})._wait_cua_ready(Unreachable())


def test_flavor_is_checked_before_creating_a_job():
    provider = HfSandboxProvider({"flavor": "cpu-upgrade"})
    provider._validate_resources(SandboxSpec(snapshot="cpu-free-ubuntu", vcpus=8, memory_gb=32))
    with pytest.raises(ValueError, match="requires 16 vCPU"):
        provider._validate_resources(
            SandboxSpec(snapshot="cpu-free-ubuntu", vcpus=16, memory_gb=64)
        )
    with pytest.raises(ValueError, match="requires 16 vCPU"):
        provider._validate_resources(
            SandboxSpec(snapshot="cpu-free-ubuntu", machine_type="c4-standard-16")
        )
    HfSandboxProvider({"flavor": "cpu-xl"})._validate_resources(
        SandboxSpec(snapshot="cpu-free-ubuntu", vcpus=16, memory_gb=64)
    )
    with pytest.raises(ValueError, match="CPU tasks only"):
        provider._validate_resources(SandboxSpec(snapshot="gpu-free"))
    with pytest.raises(ValueError, match="requires 16 vCPU"):
        provider._validate_resources(
            SandboxSpec(snapshot="cpu-free-ubuntu", machine_type="c4-standard-16", vcpus=4)
        )


async def test_release_cancels_job_even_if_proxy_stop_fails():
    class BrokenProxy:
        async def stop(self):
            raise RuntimeError("proxy close failed")

    cancelled = []
    provider = HfSandboxProvider({})
    provider._live["job-1"] = _Live(proxy=BrokenProxy(), kill=lambda: cancelled.append(True))
    with pytest.raises(RuntimeError, match="proxy close failed"):
        await provider.release(SimpleNamespace(id="job-1", metadata={}))
    assert cancelled == [True]
    assert "job-1" not in provider._live


async def test_failed_cancel_keeps_job_available_for_retry():
    class Proxy:
        async def stop(self):
            return None

    def fail_cancel():
        raise ConnectionError("cancel failed")

    provider = HfSandboxProvider({})
    provider._live["job-1"] = _Live(proxy=Proxy(), kill=fail_cancel)
    with pytest.raises(ConnectionError, match="cancel failed"):
        await provider.release(SimpleNamespace(id="job-1", metadata={}))
    assert "job-1" in provider._live
