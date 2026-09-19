# Codex with FreeRouter

The repository includes a Windows launcher for using Codex CLI with FreeRouter's
OpenAI-compatible Responses endpoint without modifying the normal Codex configuration.

```powershell
.\codex-freerouter.bat start
```

That command starts Codex with `model = "auto"`, provider `freerouter`, and
`http://127.0.0.1:8000/v1` by default. Set `FREEROUTER_BASE_URL` in the launching shell to use
another FreeRouter instance. The launcher supplies a temporary `FREEROUTER_API_KEY=sk-local` only
when the child process does not already have one; it restores the parent process environment after
Codex exits.

Return to the ordinary Codex path with:

```powershell
.\codex-freerouter.bat normal
```

`normal` invokes `codex` without provider or model overrides. Plain `codex` remains the normal
command as well. `status` runs read-only configuration probes for both paths:

```powershell
.\codex-freerouter.bat status
```

The launcher uses per-process `-c` overrides and does not edit `~/.codex/config.toml`, create
profile files, change authentication files, or persist an on/off state. FreeRouter's `auto` mode is
therefore additive at the command level. Native OpenAI models remain available through the normal
Codex command. They are not placed in the FreeRouter provider's model catalog because Codex binds a
launch to one provider; mixing native model entries into that catalog could route a normal-looking
selection to FreeRouter instead of OpenAI.

`status` exits non-zero if either path is rejected. If it reports that the normal path is rejected,
that is an existing Codex installation/configuration problem; this launcher has not changed the
file. Resolve the CLI/config version mismatch separately before relying on `normal` for a session.

Pass ordinary Codex flags after `--`:

```powershell
.\codex-freerouter.bat start -- --no-alt-screen
```

The launcher rejects extra `--model`, `--profile`, `--oss`, and provider-related `-c` overrides so
the command cannot silently start in a different backend.
