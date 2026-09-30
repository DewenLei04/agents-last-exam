"""``task_data_source: mounted:<dir>`` — cp-based staging inside the sandbox."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments import task_data as td
from ale_run.environments.task_data import baked_in_sandbox, mounted
from scripts.prepare_hf_task_data import prepare


@dataclass
class _FakeSandbox:
    """Records the sandbox I/O the backend performs; pretends `existing` paths exist."""
    existing: set[str]
    os: str = "linux"
    task_data_root: str = "/media/user/data/agenthle"
    cmds: list[str] = field(default_factory=list)
    mkdirs: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    rc: int = 0

    @property
    def is_linux(self):
        return self.os == "linux"

    async def exists(self, p):
        return p in self.existing

    async def mkdir(self, p):
        self.mkdirs.append(p)

    async def rm(self, ps):
        self.removed.extend(ps)

    async def run_command(self, cmd, *, timeout=60):
        self.cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, self.rc, "", "boom" if self.rc else "")

    async def list_dir(self, path):
        return [{"relpath": "result.txt"}]


def _spec():
    return TaskDataSpec(domain_name="demo", task_name="hello", variant_name="base")


def test_select_dispatches_mounted():
    assert td.select("mounted:/mnt/ale-task-data") is mounted


async def test_stage_input_copies_input_and_software():
    sb = _FakeSandbox(existing={
        "/mnt/d/demo/hello/base/input", "/mnt/d/demo/hello/base/software",
        "/mnt/d/demo/hello/base/reference.absent",
    })
    rep = await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d/")
    assert rep == {"staged": ["input", "software"], "source": "mounted"}
    base = "/media/user/data/agenthle/demo/hello/base"
    assert f"{base}/input" in sb.mkdirs and f"{base}/output" in sb.mkdirs and f"{base}/software" in sb.mkdirs
    assert sb.cmds[0] == f"cp -a /mnt/d/demo/hello/base/input/. {base}/input/"
    assert sb.cmds[1] == f"cp -a /mnt/d/demo/hello/base/software/. {base}/software/"
    assert sb.cmds[2].startswith(f"find {base}/software -type f -exec chmod +x")


async def test_stage_input_without_software_and_missing_input():
    sb = _FakeSandbox(existing={
        "/mnt/d/demo/hello/base/input", "/mnt/d/demo/hello/base/reference.absent",
    })
    rep = await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d")
    assert rep["staged"] == ["input"] and len(sb.cmds) == 1
    with pytest.raises(RuntimeError, match="expected input/"):
        await mounted.stage_input(_FakeSandbox(existing=set()), _spec(), source="mounted:/mnt/d")


async def test_stage_reference_after_agent_and_skip(monkeypatch):
    monkeypatch.setenv(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV, "private-test-key")
    sb = _FakeSandbox(existing={
        "/mnt/d/demo/hello/base/reference.7z",
        "/mnt/d/demo/hello/base/reference.required",
    })
    rep = await mounted.stage_reference(sb, _spec(), source="mounted:/mnt/d")
    ref = "/media/user/data/agenthle/demo/hello/base/reference"
    assert rep == {"staged": ["reference"], "source": "mounted",
                   "decrypted_from": "reference.7z"}
    assert sb.removed[:2] == [ref, "/media/user/data/agenthle/demo/hello/base/.reference_extract"]
    assert "private-test-key" in sb.cmds[0]
    assert "/mnt/d/demo/hello/base/reference.7z" in sb.cmds[0]
    assert all("cp -a" not in command for command in sb.cmds)
    assert (await mounted.stage_reference(_FakeSandbox(existing={
        "/mnt/d/demo/hello/base/reference.absent",
    }), _spec(), source="mounted:/mnt/d")) == {
        "skipped": True, "reason": "no_reference_7z"}


async def test_plaintext_mount_and_bad_password_fail_closed(monkeypatch):
    plaintext = "/mnt/d/demo/hello/base/reference"
    sb = _FakeSandbox(existing={plaintext, "/mnt/d/demo/hello/base/input"})
    with pytest.raises(RuntimeError, match="plaintext reference"):
        await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d")
    with pytest.raises(ValueError, match="plaintext reference"):
        await mounted.stage_reference(sb, _spec(), source="mounted:/mnt/d")

    archive = "/mnt/d/demo/hello/base/reference.7z"
    marker = "/mnt/d/demo/hello/base/reference.required"
    sb = _FakeSandbox(existing={archive, marker}, rc=1)
    monkeypatch.setenv(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV, "wrong-key")
    with pytest.raises(ValueError, match="7z decrypt"):
        await mounted.stage_reference(sb, _spec(), source="mounted:/mnt/d")

    sb = _FakeSandbox(existing={archive, marker})
    monkeypatch.delenv(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV)
    with pytest.raises(ValueError, match=mounted.REFERENCE_ARCHIVE_PASSWORD_ENV):
        await mounted.stage_reference(sb, _spec(), source="mounted:/mnt/d")


async def test_required_reference_must_exist_before_agent_runs():
    spec = _spec()
    sb = _FakeSandbox(existing={
        "/mnt/d/demo/hello/base/input",
        "/mnt/d/demo/hello/base/reference.required",
    })
    with pytest.raises(ValueError, match="incomplete reference state"):
        await mounted.stage_input(sb, spec, source="mounted:/mnt/d")
    assert sb.cmds == []
    with pytest.raises(ValueError, match="incomplete reference state"):
        await mounted.stage_reference(sb, spec, source="mounted:/mnt/d")
    sb = _FakeSandbox(existing={"/mnt/d/demo/hello/base/input"})
    with pytest.raises(ValueError, match="incomplete reference state"):
        await mounted.stage_input(sb, spec, source="mounted:/mnt/d")


async def test_baked_reference_still_uses_existing_password(monkeypatch):
    archive = "/media/user/data/agenthle/demo/hello/base/reference.7z"
    sb = _FakeSandbox(existing={archive})
    monkeypatch.setenv(baked_in_sandbox.REFERENCE_ARCHIVE_PASSWORD_ENV, "existing-baked-key")
    report = await baked_in_sandbox.stage_reference(sb, _spec(), source="baked_in_sandbox")
    assert report == {"staged": ["reference"], "source": "baked_in_sandbox",
                      "decrypted_from": "reference.7z"}
    assert "existing-baked-key" in sb.cmds[0]


async def test_cp_failure_raises():
    sb = _FakeSandbox(existing={
        "/mnt/d/demo/hello/base/input", "/mnt/d/demo/hello/base/reference.absent",
    }, rc=1)
    with pytest.raises(RuntimeError, match="cp .* failed"):
        await mounted.stage_input(sb, _spec(), source="mounted:/mnt/d")


def test_config_loader_accepts_mounted_source(tmp_path):
    from ale_run.orchestration.config_loader import load_experiment
    (tmp_path / "env.yaml").write_text(
        "provider: hfsandbox\ntask_data_source: mounted:/mnt/ale-task-data\noutput_path: local\n"
    )
    (tmp_path / "agent.yaml").write_text("harness: dummy\nexecutor: local\nmodel: none\n")
    (tmp_path / "exp.yaml").write_text(
        "name: t\nagents: [agent.yaml]\nenvironment: env.yaml\ntasks:\n  - path: demo/hello\n"
    )
    spec = load_experiment(tmp_path / "exp.yaml")
    assert spec.artifacts.task_data_source == "mounted:/mnt/ale-task-data"


async def test_real_mounted_reference_decrypts_after_input_stage(tmp_path, monkeypatch):
    if shutil.which("7z") is None:
        pytest.skip("7z is not installed")

    source = tmp_path / "source"
    variant = source / "demo" / "hello" / "base"
    (variant / "input").mkdir(parents=True)
    (variant / "input" / "question.txt").write_text("question")
    (variant / "reference").mkdir()
    (variant / "reference" / "answer.txt").write_text("secret answer")
    prepared = tmp_path / "prepared"
    prepare(source, prepared, password="private-test-key")

    class LocalSandbox:
        os = "linux"
        task_data_root = str(tmp_path / "sandbox")

        @property
        def is_linux(self):
            return True

        async def exists(self, path):
            return Path(path).exists()

        async def mkdir(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

        async def rm(self, paths):
            for path in paths:
                p = Path(path)
                if p.is_dir():
                    shutil.rmtree(p)
                elif p.exists():
                    p.unlink()

        async def run_command(self, command, *, timeout=60):
            return await asyncio.to_thread(
                subprocess.run, command, shell=True, text=True, capture_output=True,
                check=False, timeout=timeout,
            )

        async def list_dir(self, path):
            root = Path(path)
            return [{"relpath": str(p.relative_to(root))} for p in root.rglob("*")]

    sandbox = LocalSandbox()
    spec = _spec()
    spec.reference_dir = str(tmp_path / "sandbox" / "demo" / "hello" / "base" / "reference")
    source_uri = f"mounted:{prepared}"
    monkeypatch.setenv(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV, "private-test-key")
    await mounted.stage_input(sandbox, spec, source=source_uri)
    target = tmp_path / "sandbox" / "demo" / "hello" / "base"
    assert (target / "input" / "question.txt").read_text() == "question"
    assert not (target / "reference").exists()
    await mounted.stage_reference(sandbox, spec, source=source_uri)
    assert (target / "reference" / "answer.txt").read_text() == "secret answer"

    monkeypatch.setenv(mounted.REFERENCE_ARCHIVE_PASSWORD_ENV, "wrong-key")
    with pytest.raises(ValueError, match="7z decrypt"):
        await mounted.stage_reference(sandbox, spec, source=source_uri)
    assert not (target / "reference").exists()
