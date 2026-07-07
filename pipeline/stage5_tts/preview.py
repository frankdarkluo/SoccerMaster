"""IDE-friendly preview links and HTML player for stage 5 outputs."""
from __future__ import annotations

from pathlib import Path


def _osc8_link(path: Path, label: str | None = None) -> str:
    """OSC-8 hyperlink — Ctrl/Cmd+Click works in VS Code / Cursor terminal."""
    resolved = path.resolve()
    return f"\033]8;;{resolved.as_uri()}\033\\{label or resolved}\033]8;;\033\\"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_preview_html(
    output_dir: Path,
    video: Path,
    audio: Path,
    title: str,
    html_name: str,
) -> Path:
    """Write a self-contained HTML page with <video> and <audio> players."""
    html_path = Path(output_dir) / html_name
    html_path.write_text(
        f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
    video, audio {{ width: 100%; margin: 0.5rem 0 1.5rem; }}
    h2 {{ color: #333; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <h2>画面</h2>
  <video controls src="{video.name}"></video>
  <h2>音轨</h2>
  <audio controls src="{audio.name}"></audio>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html_path


def print_preview_links(
    video: Path,
    audio: Path,
    preview_html: Path,
    workspace_root: Path | None = None,
) -> None:
    """Print clickable terminal links for video, audio, and the HTML preview page."""
    root = (workspace_root or Path.cwd()).resolve()
    video = video.resolve()
    audio = audio.resolve()
    preview_html = preview_html.resolve()

    print(
        "\n".join([
            "",
            "── 预览（Ctrl/Cmd+Click 在 Cursor / VS Code 中打开）──",
            f"  画面    {_osc8_link(video, _rel(video, root))}",
            f"  音轨    {_osc8_link(audio, _rel(audio, root))}",
            f"  预览页  {_osc8_link(preview_html, _rel(preview_html, root))}",
            "  提示：打开预览页可在浏览器面板内直接播放画面和音轨",
            "",
        ])
    )
