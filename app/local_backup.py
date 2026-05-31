from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

SECRET_KEY_PARTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_env_without_secrets(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if any(part in key.upper() for part in SECRET_KEY_PARTS):
            continue
        values[key] = value.strip()
    return values


def export_backup(output: Path | None = None) -> Path:
    root = _project_root()
    data_dir = root / "data"
    backup_dir = root / "backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output or backup_dir / f"freerouter-local-state-{timestamp}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project": "FreeRouter",
        "contains_secrets": False,
        "files": [],
    }

    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (
            data_dir / "model_catalog.json",
            data_dir / "gateway.sqlite3",
        ):
            if path.exists():
                archive.write(path, path.relative_to(root).as_posix())
                manifest["files"].append(path.relative_to(root).as_posix())

        env_values = _read_env_without_secrets(root / ".env")
        archive.writestr("config/local-settings-without-secrets.json", json.dumps(env_values, indent=2))
        manifest["files"].append("config/local-settings-without-secrets.json")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))

    return target


def import_backup(backup_path: Path, *, overwrite: bool = False) -> list[Path]:
    root = _project_root()
    restored: list[Path] = []
    allowed = {
        "data/model_catalog.json": root / "data" / "model_catalog.json",
        "data/gateway.sqlite3": root / "data" / "gateway.sqlite3",
    }

    with zipfile.ZipFile(backup_path) as archive:
        names = set(archive.namelist())
        for archive_name, target in allowed.items():
            if archive_name not in names:
                continue
            if target.exists() and not overwrite:
                raise FileExistsError(
                    f"{target} already exists. Re-run with --overwrite to replace local state."
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(archive_name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            restored.append(target)

    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description="Export or import FreeRouter local state.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Create a local state backup ZIP.")
    export_parser.add_argument("--output", type=Path, help="Backup ZIP path.")

    import_parser = subparsers.add_parser("import", help="Restore local state from a backup ZIP.")
    import_parser.add_argument("backup", type=Path)
    import_parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()
    if args.command == "export":
        target = export_backup(args.output)
        print(f"Exported local state to {target}")
    elif args.command == "import":
        restored = import_backup(args.backup, overwrite=args.overwrite)
        if restored:
            print("Restored:")
            for path in restored:
                print(f"  {path}")
        else:
            print("No restorable local state files found in backup.")


if __name__ == "__main__":
    main()
