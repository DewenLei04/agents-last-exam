"""Check the published ALE image, private mount, and scoring data on HF Jobs.

Requires ``hf auth login`` and Hugging Face Jobs credits. The provider cancels
the Job on startup failure; this script releases it after the checks. Requires
``7z`` on the host to create the synthetic encrypted reference archive.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from pathlib import Path

from huggingface_hub import HfApi

from ale_run.base_interface import SandboxSpec, TaskDataSpec
from ale_run.environments.providers.hfsandbox import HfSandboxProvider
from ale_run.environments.task_data import mounted, task_subdir
from scripts.prepare_hf_task_data import prepare

logger = logging.getLogger(__name__)


async def main() -> None:
    api = HfApi()
    bucket_id = f"{api.whoami()['name']}/ale-pr100-official-smoke-{uuid.uuid4().hex[:8]}"
    password = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix="ale-hf-official-") as temp_dir:
        root = Path(temp_dir)
        variant = root / "source" / "demo" / "hello" / "base"
        (variant / "input").mkdir(parents=True)
        (variant / "reference").mkdir()
        (variant / "input" / "order.json").write_text('{"smoke": true}\n')
        (variant / "reference" / "result.txt").write_text("ALE_REFERENCE_OK\n")
        prepared = root / "prepared"
        prepare(root / "source", prepared, password=password)
        api.create_bucket(bucket_id, private=True)
        try:
            api.sync_bucket(str(prepared), f"hf://buckets/{bucket_id}")
            await check_job(bucket_id, password)
        finally:
            api.delete_bucket(bucket_id, missing_ok=True)
            logger.info("Deleted synthetic bucket %s", bucket_id)


async def check_job(bucket_id: str, password: str) -> None:
    provider = HfSandboxProvider({
        "image": "ale-ubuntu22-docker",
        "flavor": "cpu-xl",
        "job_timeout": "25m",
        "start_timeout": 1200,
        "cua_ready_timeout": 120,
        "volumes": [f"hf://buckets/{bucket_id}:/mnt/ale-task-data:ro"],
    })
    sandbox = await provider.acquire(SandboxSpec(
        snapshot="cpu-free-ubuntu", vcpus=4, memory_gb=16,
        task_id="hf-official-image-smoke",
    ))
    try:
        result = await sandbox.run_command("printf ALE_OFFICIAL_IMAGE_OK", timeout=30)
        if result.returncode != 0 or result.stdout.strip() != "ALE_OFFICIAL_IMAGE_OK":
            raise RuntimeError(f"ALE image command failed: {result!r}")
        task_data = TaskDataSpec(
            requires_task_data=True,
            domain_name="demo", task_name="hello", variant_name="base",
        )
        source = "mounted:/mnt/ale-task-data"
        await mounted.stage_input(sandbox, task_data, source=source)
        base = task_subdir(sandbox, task_data)
        if await sandbox.read_text(f"{base}/input/order.json") != '{"smoke": true}\n':
            raise RuntimeError("Mounted input content mismatch")
        if await sandbox.exists(f"{base}/reference"):
            raise RuntimeError("Reference visible before scoring phase")
        env_check = await sandbox.run_command(
            'test -z "${ALE_MOUNTED_REFERENCE_ARCHIVE_PASSWORD:-}"', timeout=30,
        )
        if env_check.returncode != 0:
            raise RuntimeError("Reference password leaked into Job environment")
        await sandbox.write_file(f"{base}/output/result.txt", "ALE_REFERENCE_OK\n")
        old_password = os.environ.get(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV)
        os.environ[mounted.REFERENCE_ARCHIVE_PASSWORD_ENV] = password
        try:
            await mounted.stage_reference(sandbox, task_data, source=source)
        finally:
            if old_password is None:
                os.environ.pop(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV, None)
            else:
                os.environ[mounted.REFERENCE_ARCHIVE_PASSWORD_ENV] = old_password
        expected = (await sandbox.read_text(f"{base}/reference/result.txt")).strip()
        actual = (await sandbox.read_text(f"{base}/output/result.txt")).strip()
        if expected != actual or expected != "ALE_REFERENCE_OK":
            raise RuntimeError(f"Synthetic score mismatch: {actual!r} vs {expected!r}")
        print(f"Published ALE image and private scoring data passed: {sandbox.id}")
    finally:
        await provider.release(sandbox)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
