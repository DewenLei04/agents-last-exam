"""Prepare an ALE task-data tree for a read-only HF Jobs volume.

The destination contains input/, software/, and an explicit reference state.
Run on a trusted host with 7z installed. The destination must not exist yet.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

PASSWORD_ENV = "ALE_MOUNTED_REFERENCE_ARCHIVE_PASSWORD"


def _variants(root: Path):
    for domain in sorted(root.iterdir()):
        if not domain.is_dir() or domain.is_symlink():
            raise ValueError(f"expected domain directory: {domain}")
        for task in sorted(domain.iterdir()):
            if not task.is_dir() or task.is_symlink():
                raise ValueError(f"expected task directory: {task}")
            for variant in sorted(task.iterdir()):
                if not variant.is_dir() or variant.is_symlink():
                    raise ValueError(f"expected variant directory: {variant}")
                yield variant.relative_to(root), variant


def _reject_links(path: Path) -> None:
    if any(item.is_symlink() for item in path.rglob("*")):
        raise ValueError(f"symlinks are not allowed in HF task data: {path}")


def validate_prepared(root: Path) -> int:
    """Fail if any variant contains plaintext reference material."""
    count = 0
    for _, variant in _variants(root):
        allowed = {"input", "software", "reference.7z", "reference.required", "reference.absent"}
        extra = {item.name for item in variant.iterdir()} - allowed
        if extra:
            raise ValueError(f"unexpected data in {variant}: {sorted(extra)}")
        if not (variant / "input").is_dir():
            raise ValueError(f"missing input/: {variant}")
        if (variant / "software").exists() and not (variant / "software").is_dir():
            raise ValueError(f"software/ must be a directory: {variant}")
        if (variant / "reference.7z").exists() and not (variant / "reference.7z").is_file():
            raise ValueError(f"reference.7z must be a file: {variant}")
        marker = variant / "reference.required"
        absent = variant / "reference.absent"
        archive_exists = (variant / "reference.7z").exists()
        if marker.exists() == absent.exists() or marker.exists() != archive_exists:
            raise ValueError(f"incomplete reference state: {variant}")
        if marker.exists() and marker.read_text() != "reference.7z\n":
            raise ValueError(f"invalid reference.required marker: {variant}")
        if absent.exists() and absent.read_text() != "none\n":
            raise ValueError(f"invalid reference.absent marker: {variant}")
        _reject_links(variant)
        count += 1
    return count


def prepare(source: Path, destination: Path, *, password: str) -> int:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("destination must be outside the source tree")
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    seven_zip = shutil.which("7z")
    if not seven_zip:
        raise RuntimeError("7z is required to encrypt reference archives")
    if not password:
        raise ValueError(f"{PASSWORD_ENV} must contain a private password")
    if "\n" in password or "\r" in password:
        raise ValueError(f"{PASSWORD_ENV} must not contain a newline")

    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        for rel, variant in _variants(source):
            out = staging / rel
            out.mkdir(parents=True)
            allowed = {"input", "software", "reference"}
            extra = {item.name for item in variant.iterdir()} - allowed
            if extra:
                raise ValueError(f"unexpected source data in {variant}: {sorted(extra)}")
            if not (variant / "input").is_dir():
                raise ValueError(f"missing input/: {variant}")
            for name in ("software", "reference"):
                if (variant / name).exists() and not (variant / name).is_dir():
                    raise ValueError(f"{name}/ must be a directory: {variant}")
            _reject_links(variant)
            for name in ("input", "software"):
                if (variant / name).is_dir():
                    shutil.copytree(variant / name, out / name)
            reference = variant / "reference"
            if reference.is_dir():
                try:
                    subprocess.run(
                        [seven_zip, "a", "-t7z", "-mhe=on", "-p",
                         str(out / "reference.7z"), "."],
                        cwd=reference, input=password + "\n", text=True,
                        check=True, capture_output=True,
                    )
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f"7z failed to encrypt {variant} (rc={exc.returncode})"
                    ) from None
                if not (out / "reference.7z").is_file():
                    raise RuntimeError(f"7z did not create an archive for {variant}")
                (out / "reference.required").write_text("reference.7z\n")
            else:
                (out / "reference.absent").write_text("none\n")
        count = validate_prepared(staging)
        staging.rename(destination)
        return count
    except BaseException:
        shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="extracted ALE task-data directory")
    parser.add_argument("destination", type=Path, help="new upload-ready directory")
    args = parser.parse_args()
    count = prepare(args.source, args.destination, password=os.environ.get(PASSWORD_ENV, ""))
    print(f"Prepared {count} variants in {args.destination}")


if __name__ == "__main__":
    main()
