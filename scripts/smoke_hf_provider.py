"""Check the HF Jobs provider's CUA transport with a small AMD64 image.

Requires ``hf auth login`` and a positive Hugging Face credit balance.
This image does not contain ALE task software or reference data.
"""
from __future__ import annotations

import asyncio

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers.hfsandbox import HfSandboxProvider


async def main() -> None:
    provider = HfSandboxProvider({
        "image": "ale-kasm",
        "image_ref": (
            "trycua/cua-ubuntu@sha256:"
            "2bb539bd42f59f9e2a889faa4ebcb535a0c06397a3b98e45fd5cc8a96c22014a"
        ),
        "job_command": [
            "/bin/bash", "-lc",
            (
                "Xvfb :0 -screen 0 1024x768x24 >/tmp/xvfb.log 2>&1 & "
                "export DISPLAY=:0 CUA_TELEMETRY_DISABLED=1; "
                "exec python3 -m computer_server --host 0.0.0.0 --port 8000"
            ),
        ],
        "flavor": "cpu-basic",
        "job_timeout": "5m",
        "start_timeout": 180,
        "cua_ready_timeout": 90,
    })
    sandbox = await provider.acquire(SandboxSpec(
        snapshot="cpu-free-ubuntu", vcpus=2, memory_gb=16,
        task_id="hf-provider-smoke",
    ))
    try:
        result = await sandbox.run_command("printf ALE_HF_PROVIDER_OK", timeout=30)
        if result.returncode != 0 or result.stdout.strip() != "ALE_HF_PROVIDER_OK":
            raise RuntimeError(f"CUA command failed: {result!r}")
        print(f"HF provider smoke passed: {sandbox.id}")
    finally:
        await provider.release(sandbox)


if __name__ == "__main__":
    asyncio.run(main())
