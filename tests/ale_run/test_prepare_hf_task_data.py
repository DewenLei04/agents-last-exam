"""The HF upload tree must never contain plaintext reference files."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.prepare_hf_task_data import prepare, validate_prepared


def test_prepare_encrypts_references_and_rejects_plaintext(tmp_path: Path):
    source = tmp_path / "source"
    variant = source / "demo" / "hello" / "base"
    (variant / "input").mkdir(parents=True)
    (variant / "input" / "question.txt").write_text("question")
    (variant / "reference").mkdir()
    (variant / "reference" / "answer.txt").write_text("secret answer")

    def fake_seven_zip(argv, *, cwd, input, text, check, capture_output):
        assert cwd == variant / "reference"
        assert "-mhe=on" in argv
        assert "-p" in argv and "private-test-key" not in " ".join(argv)
        assert input == "private-test-key\n" and text is True
        Path(argv[-2]).write_bytes(b"encrypted archive placeholder")

    destination = tmp_path / "prepared"
    with patch("scripts.prepare_hf_task_data.shutil.which", return_value="/usr/bin/7z"), patch(
        "scripts.prepare_hf_task_data.subprocess.run", side_effect=fake_seven_zip
    ):
        assert prepare(source, destination, password="private-test-key") == 1
    prepared = destination / "demo" / "hello" / "base"
    assert (prepared / "input" / "question.txt").read_text() == "question"
    assert (prepared / "reference.7z").is_file()
    assert (prepared / "reference.required").read_text() == "reference.7z\n"
    assert not (prepared / "reference.absent").exists()
    assert not (prepared / "reference").exists()
    assert validate_prepared(destination) == 1
    (prepared / "reference").mkdir()
    with pytest.raises(ValueError, match="unexpected data"):
        validate_prepared(destination)


def test_prepare_marks_variants_without_reference(tmp_path: Path):
    source = tmp_path / "source"
    variant = source / "demo" / "hello" / "base"
    (variant / "input").mkdir(parents=True)
    with patch("scripts.prepare_hf_task_data.shutil.which", return_value="/usr/bin/7z"):
        assert prepare(source, tmp_path / "prepared", password="private-test-key") == 1
    prepared = tmp_path / "prepared" / "demo" / "hello" / "base"
    assert (prepared / "reference.absent").read_text() == "none\n"
    assert not (prepared / "reference.7z").exists()
    (prepared / "reference.absent").unlink()
    with pytest.raises(ValueError, match="incomplete reference state"):
        validate_prepared(tmp_path / "prepared")


def test_encryption_error_does_not_disclose_password(tmp_path: Path):
    source = tmp_path / "source"
    variant = source / "demo" / "hello" / "base"
    (variant / "input").mkdir(parents=True)
    (variant / "reference").mkdir()

    def fail(argv, **kwargs):
        import subprocess

        raise subprocess.CalledProcessError(2, argv, stderr=b"failure")

    with patch("scripts.prepare_hf_task_data.shutil.which", return_value="/usr/bin/7z"), patch(
        "scripts.prepare_hf_task_data.subprocess.run", side_effect=fail
    ), pytest.raises(RuntimeError, match="7z failed") as exc:
        prepare(source, tmp_path / "prepared", password="do-not-disclose-this")
    assert "do-not-disclose-this" not in str(exc.value)
    assert not (tmp_path / "prepared.staging").exists()


def test_real_7z_round_trip(tmp_path: Path):
    seven_zip = shutil.which("7z")
    if seven_zip is None:
        pytest.skip("7z is not installed")

    source = tmp_path / "source"
    reference = source / "demo" / "hello" / "base" / "reference"
    reference.mkdir(parents=True)
    (reference.parent / "input").mkdir()
    (reference / "answer.txt").write_text("private answer\n")
    (reference / "nested").mkdir()
    (reference / "nested" / "data.bin").write_bytes(b"\x00\xff\x01")

    prepared = tmp_path / "prepared"
    password = "random-test-password"
    assert prepare(source, prepared, password=password) == 1
    variant = prepared / "demo" / "hello" / "base"
    archive = variant / "reference.7z"
    assert archive.is_file() and not (variant / "reference").exists()
    assert (variant / "reference.required").is_file()

    locked = subprocess.run([seven_zip, "l", str(archive)], check=False, capture_output=True)
    assert locked.returncode != 0
    assert b"answer.txt" not in locked.stdout
    wrong = subprocess.run(
        [seven_zip, "x", "-pwrong-password", str(archive),
         f"-o{tmp_path / 'wrong'}", "-y"], check=False, capture_output=True,
    )
    assert wrong.returncode != 0

    extracted = tmp_path / "extracted"
    subprocess.run(
        [seven_zip, "x", f"-p{password}", str(archive), f"-o{extracted}", "-y"],
        check=True, capture_output=True,
    )
    assert (extracted / "answer.txt").read_text() == "private answer\n"
    assert (extracted / "nested" / "data.bin").read_bytes() == b"\x00\xff\x01"
