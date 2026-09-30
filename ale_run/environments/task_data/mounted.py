"""``task_data_source: mounted:<sandbox_dir>`` — task data already visible
*inside* the sandbox at ``<sandbox_dir>/<domain>/<task>/<variant>/`` (e.g. a
read-only HF bucket mounted into the sandbox job by the ``hfsandbox`` provider,
or any NFS/FUSE share the sandbox can see). Copies with plain ``cp`` via the
sandbox's cua-server, so it works on every provider — nothing runs on the host.

Layout expected under ``<sandbox_dir>`` is::

    <domain>/<task>/<variant>/{input,software,reference.7z,reference.required}

Variants without reference data carry a ``reference.absent`` marker instead.

Phase 1 (``stage_input``): copy ``input/`` (+ ``software/``, made executable)
into the task dir and create ``output/``. Phase 3 (``stage_reference``):
decrypt ``reference.7z`` after the agent has finished. The mounted source
must contain no plaintext ``reference/`` directories anywhere: the agent can
read the entire mount during its run.
Linux-only (the mount point is a POSIX path).
"""
from __future__ import annotations

import logging
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, shell_q, task_subdir

logger = logging.getLogger(__name__)

_PREFIX = "mounted:"
REFERENCE_ARCHIVE_PASSWORD_ENV = "ALE_MOUNTED_REFERENCE_ARCHIVE_PASSWORD"


def _mounted_task_dir(source: str, task_data: TaskDataSpec) -> str:
    root = source[len(_PREFIX):].rstrip("/")
    if not root:
        raise ValueError("task_data_source=mounted: expected 'mounted:<sandbox_dir>'")
    return "/".join([
        root, task_data.domain_name or "", task_data.task_name or "",
        task_data.variant_name or "",
    ])


async def _copy_tree(sandbox: SandboxHandle, src: str, dst: str, *, timeout: float) -> None:
    """``cp -a src/. dst/`` — copies CONTENTS into dst whether or not it exists."""
    await sandbox.mkdir(dst)
    r = await sandbox.run_command(
        f"cp -a {shell_q(sandbox, src)}/. {shell_q(sandbox, dst)}/", timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"task_data_source=mounted: cp {src} -> {dst} failed (rc={r.returncode}): "
            f"{(r.stderr or r.stdout or '')[:300]}"
        )


async def _has_reference_archive(sandbox: SandboxHandle, src: str) -> bool:
    archive = await sandbox.exists(f"{src}/reference.7z")
    required = await sandbox.exists(f"{src}/reference.required")
    absent = await sandbox.exists(f"{src}/reference.absent")
    if required == absent or archive != required:
        raise ValueError(f"task_data_source=mounted: incomplete reference state at {src!r}")
    return required


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Copy input/ (+ software/) from the mount into the task dir; make output/."""
    if not sandbox.is_linux:
        raise NotImplementedError("task_data_source=mounted is Linux-only")
    src = _mounted_task_dir(source, task_data)
    if await sandbox.exists(f"{src}/reference"):
        raise RuntimeError(
            f"task_data_source=mounted: plaintext reference/ is visible at {src!r}; "
            "publish only encrypted reference.7z archives"
        )
    if not await sandbox.exists(f"{src}/input"):
        raise RuntimeError(
            f"task_data_source=mounted: expected input/ at {src!r} inside the sandbox, "
            f"not found. Is the task-data volume mounted at {source[len(_PREFIX):]!r}?"
        )
    await _has_reference_archive(sandbox, src)
    base = task_subdir(sandbox, task_data)
    await _copy_tree(sandbox, f"{src}/input", join(sandbox, base, "input"), timeout=1800)
    await sandbox.mkdir(join(sandbox, base, "output"))
    staged = ["input"]
    if await sandbox.exists(f"{src}/software"):
        sw_dst = join(sandbox, base, "software")
        await _copy_tree(sandbox, f"{src}/software", sw_dst, timeout=1800)
        # mirror baked_in_sandbox / local: software wrappers must be executable.
        await sandbox.run_command(
            f"find {shell_q(sandbox, sw_dst)} -type f -exec chmod +x {{}} +", timeout=120,
        )
        staged.append("software")
    logger.info("mounted: staged %s for %s from %s", staged, task_data.task_name, src)
    return {"staged": staged, "source": "mounted"}


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    """Decrypt a mounted reference archive after the agent, before evaluation."""
    if not sandbox.is_linux:
        raise NotImplementedError("task_data_source=mounted is Linux-only")
    src = _mounted_task_dir(source, task_data)
    if await sandbox.exists(f"{src}/reference"):
        raise ValueError(f"task_data_source=mounted: plaintext reference/ at {src!r}")
    archive = f"{src}/reference.7z"
    if not await _has_reference_archive(sandbox, src):
        return {"skipped": True, "reason": "no_reference_7z"}
    base = task_subdir(sandbox, task_data)
    target = join(sandbox, base, "reference")
    from .baked_in_sandbox import decrypt_reference_archive

    await decrypt_reference_archive(
        sandbox, archive=archive, target=target,
        password_env=REFERENCE_ARCHIVE_PASSWORD_ENV,
    )
    return {"staged": ["reference"], "source": "mounted",
            "decrypted_from": "reference.7z"}
