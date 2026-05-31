from __future__ import annotations

import json
import zipfile

import app.local_backup as local_backup


def test_export_backup_omits_secret_env_values(tmp_path, monkeypatch):
    root = tmp_path
    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "model_catalog.json").write_text("[]\n", encoding="utf-8")
    (data_dir / "gateway.sqlite3").write_bytes(b"sqlite")
    (root / ".env").write_text(
        "GROQ_API_KEY=secret\nREQUEST_TIMEOUT_SECONDS=10\nCUSTOM_TOKEN=secret\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(local_backup, "_project_root", lambda: root)

    target = local_backup.export_backup(root / "state.zip")

    with zipfile.ZipFile(target) as archive:
        settings = json.loads(archive.read("config/local-settings-without-secrets.json"))
        names = set(archive.namelist())

    assert "data/model_catalog.json" in names
    assert "data/gateway.sqlite3" in names
    assert settings == {"REQUEST_TIMEOUT_SECONDS": "10"}


def test_import_backup_refuses_to_overwrite_by_default(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "data").mkdir()
    (root / "data" / "model_catalog.json").write_text("[]\n", encoding="utf-8")
    backup = tmp_path / "state.zip"
    with zipfile.ZipFile(backup, "w") as archive:
        archive.writestr("data/model_catalog.json", "[{}]\n")
    monkeypatch.setattr(local_backup, "_project_root", lambda: root)

    try:
        local_backup.import_backup(backup)
    except FileExistsError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected import to refuse overwriting existing state")


def test_import_backup_can_overwrite(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    backup = tmp_path / "state.zip"
    with zipfile.ZipFile(backup, "w") as archive:
        archive.writestr("data/model_catalog.json", "[{}]\n")
    monkeypatch.setattr(local_backup, "_project_root", lambda: root)

    restored = local_backup.import_backup(backup, overwrite=True)

    assert restored == [root / "data" / "model_catalog.json"]
    assert (root / "data" / "model_catalog.json").read_text(encoding="utf-8") == "[{}]\n"
