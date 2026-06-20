from pathlib import Path


def test_sidecar_build_excludes_non_importable_python_filenames() -> None:
    script = Path("scripts/build-sidecar.ps1").read_text(encoding="utf-8")

    assert "InvalidModuleFiles" in script
    assert "'^[A-Za-z_][A-Za-z0-9_]*$'" in script
    assert "Excluding non-importable Python filenames from the sidecar build" in script
    assert '$PyInstallerArgs += @("--exclude-module", $InvalidModuleName)' in script
    assert '"--noconsole"' in script
