"""Shell execution tool."""

import asyncio
import os
import re
import shlex
from pathlib import Path
from typing import Any

from fubot.agent.tools.base import Tool
from fubot.agent.tools.path_safety import resolve_user_path, resolve_within_directory


class ExecTool(Tool):
    """Tool to execute shell commands."""

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
        path_append: str = "",
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",          # rm -r, rm -rf, rm -fr
            r"\bdel\s+/[fq]\b",              # del /f, del /q
            r"\brmdir\s+/s\b",               # rmdir /s
            r"(?:^|[;&|]\s*)format\b",       # format (as standalone command only)
            r"\b(mkfs|diskpart)\b",          # disk operations
            r"\bdd\s+if=",                   # dd
            r">\s*/dev/sd",                  # write to disk
            r"\b(shutdown|reboot|poweroff)\b",  # system power
            r":\(\)\s*\{.*\};\s*:",          # fork bomb
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace
        self.path_append = path_append

    @property
    def name(self) -> str:
        return "exec"

    _MAX_TIMEOUT = 600
    _MAX_OUTPUT = 10_000

    @property
    def description(self) -> str:
        return "Execute a shell command and return its output. Use with caution."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional working directory for the command",
                },
                "timeout": {
                    "type": "integer",
                    "description": (
                        "Timeout in seconds. Increase for long-running commands "
                        "like compilation or installation (default 60, max 600)."
                    ),
                    "minimum": 1,
                    "maximum": 600,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None,
        timeout: int | None = None, **kwargs: Any,
    ) -> str:
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        env = os.environ.copy()
        if self.path_append:
            env["PATH"] = env.get("PATH", "") + os.pathsep + self.path_append

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    return (
                        f"Error: Command timed out after {effective_timeout} seconds "
                        "and process termination could not be confirmed promptly"
                    )
                return f"Error: Command timed out after {effective_timeout} seconds"

            output_parts = []

            if stdout:
                output_parts.append(stdout.decode("utf-8", errors="replace"))

            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    output_parts.append(f"STDERR:\n{stderr_text}")

            output_parts.append(f"\nExit code: {process.returncode}")

            result = "\n".join(output_parts) if output_parts else "(no output)"

            # Head + tail truncation to preserve both start and end of output
            max_len = self._MAX_OUTPUT
            if len(result) > max_len:
                half = max_len // 2
                result = (
                    result[:half]
                    + f"\n\n... ({len(result) - max_len:,} chars truncated) ...\n\n"
                    + result[-half:]
                )

            return result

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """Best-effort safety guard for potentially destructive commands."""
        cmd = command.strip()
        lower = cmd.lower()

        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"

        if self.allow_patterns:
            if not any(re.search(p, lower) for p in self.allow_patterns):
                return "Error: Command blocked by safety guard (not in allowlist)"

        if self.restrict_to_workspace:
            if re.search(r"\b(?:python|python3|node|ruby|perl|php|bash|sh)\b\s+(?:-c|-e|-r)\b", lower):
                return "Error: Command blocked by safety guard (inline script execution is disabled with workspace restriction)"
            try:
                cwd_path = resolve_within_directory(
                    cwd,
                    allowed_dir=Path(self.working_dir or cwd),
                    label="Working directory",
                )
            except PermissionError:
                return "Error: Command blocked by safety guard (invalid working dir)"

            path_error = self._guard_workspace_paths(cmd, cwd_path)
            if path_error:
                return path_error

        return None

    def _guard_workspace_paths(self, command: str, cwd_path: Path) -> str | None:
        """Resolve referenced paths and block workspace escapes."""
        if "..\\" in command or "../" in command:
            return "Error: Command blocked by safety guard (path traversal detected)"

        candidates = set(self._extract_absolute_paths(command))
        for token in self._extract_path_tokens(command, cwd_path):
            candidates.add(token)

        for raw in candidates:
            try:
                resolve_within_directory(
                    raw,
                    base_dir=cwd_path,
                    allowed_dir=Path(self.working_dir or cwd_path),
                )
            except PermissionError:
                return "Error: Command blocked by safety guard (path outside working dir)"
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)   # Windows: C:\...
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command) # POSIX: /absolute only
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command) # POSIX/Windows home shortcut: ~
        return win_paths + posix_paths + home_paths

    @staticmethod
    def _extract_path_tokens(command: str, cwd_path: Path) -> list[str]:
        try:
            tokens = shlex.split(command, posix=os.name != "nt")
        except ValueError:
            return []

        candidates: list[str] = []
        for index, token in enumerate(tokens):
            if index == 0 or not token or token.startswith("-") or "=" in token.split(os.sep, 1)[0]:
                continue
            if token in {"|", "&&", "||", ";"}:
                continue
            if (
                "/" in token
                or "\\" in token
                or token.startswith(("~", "."))
                or (cwd_path / token).exists()
            ):
                candidates.append(token)
                continue
            resolved = resolve_user_path(token, base_dir=cwd_path)
            if resolved.exists():
                candidates.append(token)
        return candidates
