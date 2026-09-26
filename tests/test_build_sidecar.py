import json
from pathlib import Path


def test_sidecar_build_excludes_non_importable_python_filenames() -> None:
    script = Path("scripts/build-sidecar.ps1").read_text(encoding="utf-8")

    assert "InvalidModuleFiles" in script
    assert "'^[A-Za-z_][A-Za-z0-9_]*$'" in script
    assert "Excluding non-importable Python filenames from the sidecar build" in script
    assert '$PyInstallerArgs += @("--exclude-module", $InvalidModuleName)' in script
    assert '"--noconsole"' in script


def test_desktop_check_bootstraps_a_host_sidecar_placeholder() -> None:
    package = json.loads(Path("package.json").read_text(encoding="utf-8"))
    script = Path("scripts/check-desktop.mjs").read_text(encoding="utf-8")

    assert package["scripts"]["check:desktop"] == "node scripts/check-desktop.mjs"
    assert 'execFileSync("rustc", ["-vV"]' in script
    assert "freerouterd-${targetTriple}${extension}" in script
    assert "unlinkSync(placeholderPath)" in script
