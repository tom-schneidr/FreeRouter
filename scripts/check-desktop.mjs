import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const manifestPath = join(projectRoot, "apps", "desktop", "src-tauri", "Cargo.toml");
const sidecarDirectory = join(projectRoot, "dist-sidecar");

function hostTargetTriple() {
  const rustInfo = execFileSync("rustc", ["-vV"], {
    cwd: projectRoot,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "inherit"],
  });
  const match = rustInfo.match(/^host:\s*(\S+)$/m);
  if (!match) {
    throw new Error("Could not determine the Rust host target triple from rustc -vV.");
  }
  return match[1];
}

const targetTriple = hostTargetTriple();
const extension = process.platform === "win32" ? ".exe" : "";
const placeholderPath = join(sidecarDirectory, `freerouterd-${targetTriple}${extension}`);
const createdPlaceholder = !existsSync(placeholderPath);

if (createdPlaceholder) {
  mkdirSync(sidecarDirectory, { recursive: true });
  writeFileSync(placeholderPath, "");
}

try {
  execFileSync("cargo", ["check", "--manifest-path", manifestPath], {
    cwd: projectRoot,
    stdio: "inherit",
  });
} finally {
  if (createdPlaceholder) {
    unlinkSync(placeholderPath);
  }
}
