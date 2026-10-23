"""Animations to GIF, WebM or MP4 (1.1.6): normal modes, optimization
trajectories, MD runs and NEB paths.

Two ways in, one encoder:

* the GUI captures the frames its own 3D view draws (Export Animation…:
  js/lazy/anim_export.js renders each frame off screen at the chosen size and
  posts it here), so the animation is exactly the view;
* ``mace animate FILE`` (and the dialog's "Rendered by MACE" source) draws the
  frames here, headless, with Pillow: a ball-and-stick picture of each frame
  of a CRYSTAL FREQ output (one normal mode, as the GUI's mode view moves it),
  of a CRYSTAL optimization (every geometry step), or of any multi-frame file
  ASE reads (an MD trajectory, a NEB path as .xyz/.extxyz/.traj).

GIF is written by Pillow (a matplotlib dependency): one adaptive palette for
the whole animation (no flicker between frames), an optional transparent
background, a loop count. WebM (VP9, with transparency) and MP4 (H.264) need
ffmpeg on PATH (or MACE_FFMPEG); without it the error says so and GIF still
works.

The grayscale print patterns (Settings › Viewport › Print patterns,
js/viewport/print_patterns.js) are drawn here in the same order and by the
same rule, so a black-and-white print of a CLI animation reads as the GUI's.
"""
from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

FORMATS = ("gif", "webm", "mp4")
BACKGROUNDS = ("light", "dark", "white", "black", "transparent")
PATTERN_MODES = ("off", "auto", "all")
# How a FREQ mode's displacement is scaled, as the GUI's freq.amplitudeScale setting: "physical" draws
# CRYSTAL's printed zero-point classical amplitude x amplitude (the GUI default since 1.1.6);
# "normalized" moves the largest atom 0.5 A x amplitude (the GUI's "Fit 0.5 A", 1.1.5's drawing).
AMPLITUDE_SCALES = ("physical", "normalized")
# js/viewport/print_patterns.js PATTERNS: the same order
PATTERNS = ("slash", "cross", "stipple", "backslash", "horizontal", "grid", "vertical", "dots")
SIMILAR = 1.35
MAX_FRAMES = 600
MAX_SIDE = 4096
MAX_FPS = 60
# every frame is held as RGBA (4 bytes a pixel) until it is encoded: frames x width x height at most this
MAX_PIXELS = 1_500_000_000
BG = {"light": (247, 248, 250, 255), "dark": (22, 25, 31, 255), "white": (255, 255, 255, 255),
      "black": (0, 0, 0, 255), "transparent": (255, 255, 255, 0)}
INK = {"light": (21, 24, 30), "dark": (231, 233, 238), "white": (21, 24, 30), "black": (231, 233, 238),
       "transparent": (21, 24, 30)}
SESSION_TTL = 3600.0          # an unfinished GUI capture is let go of after an hour


class AnimError(ValueError):
    """A request that cannot be met, said in words (the CLI prints it; the GUI shows it)."""


# ---------------------------------------------------------------- ffmpeg
def ffmpeg_path() -> Optional[str]:
    """ffmpeg from MACE_FFMPEG or PATH, or None."""
    env = os.environ.get("MACE_FFMPEG")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    return shutil.which("ffmpeg")


_ENCODERS: Dict[str, set] = {}


def _encoders(ff: str) -> set:
    if ff not in _ENCODERS:
        try:
            r = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
            names = set()
            for line in r.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] == "V":
                    names.add(parts[1])
            _ENCODERS[ff] = names
        except (OSError, subprocess.SubprocessError):
            _ENCODERS[ff] = set()
    return _ENCODERS[ff]


def capabilities() -> Dict[str, Any]:
    """What this machine can write: GIF always; WebM and MP4 with ffmpeg (which encoder)."""
    ff = ffmpeg_path()
    enc = _encoders(ff) if ff else set()
    webm = "libvpx-vp9" if "libvpx-vp9" in enc else "libvpx" if "libvpx" in enc else None
    mp4 = "libx264" if "libx264" in enc else "libopenh264" if "libopenh264" in enc else "mpeg4" if "mpeg4" in enc else None
    return {"gif": True, "webm": bool(ff and webm), "mp4": bool(ff and mp4), "ffmpeg": ff, "webm_encoder": webm,
            "mp4_encoder": mp4, "alpha_video": webm == "libvpx-vp9",
            "why": None if ff else "WebM and MP4 need ffmpeg on PATH, which this machine does not have. Install it "
                                   "(conda install -c conda-forge ffmpeg, or the system's package manager) or set "
                                   "MACE_FFMPEG; GIF needs nothing more."}


# ---------------------------------------------------------------- encoding
def _loop_args(loop: int) -> Optional[int]:
    """GIF's NETSCAPE loop value for "plays": 0 = forever; n plays = n - 1 repeats (None: play once)."""
    loop = int(loop)
    if loop <= 0:
        return 0
    return None if loop == 1 else loop - 1


def encode(frames: Sequence[Any], out: Path, fmt: str = "gif", fps: float = 20.0, loop: int = 0,
           transparent: bool = False, matte: Tuple[int, int, int] = (255, 255, 255),
           progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """Write ``frames`` (PIL images, RGB or RGBA, one size) as ``fmt`` to ``out``.

    loop: 0 plays forever; n >= 1 plays n times (GIF's loop count; a video is written n times over).
    transparent: keep the frames' alpha (GIF: one palette index; WebM VP9: yuva420p; MP4 has none:
    composited on ``matte``)."""
    from PIL import Image
    fmt = str(fmt).lower()
    if fmt not in FORMATS:
        raise AnimError(f"the format is one of {', '.join(FORMATS)}")
    if not frames:
        raise AnimError("no frames to write")
    fps = float(fps)
    if not (0.5 <= fps <= MAX_FPS):
        raise AnimError(f"the frame rate runs from 0.5 to {MAX_FPS} frames per second")
    size = frames[0].size
    frames = [f if f.size == size else f.resize(size) for f in frames]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "gif":
        _write_gif(frames, out, fps, loop, transparent, matte, progress)
    else:
        _write_video(frames, out, fmt, fps, loop, transparent, matte, progress)
    return {"path": str(out), "bytes": out.stat().st_size, "frames": len(frames), "size": list(size), "format": fmt,
            "fps": fps, "loop": int(loop)}


def _flatten(img, matte):
    from PIL import Image
    if img.mode != "RGBA":
        return img.convert("RGB")
    bg = Image.new("RGBA", img.size, tuple(matte) + (255,))
    bg.alpha_composite(img)
    return bg.convert("RGB")


def _write_gif(frames, out: Path, fps, loop, transparent, matte, progress):
    from PIL import Image
    # one palette for every frame (an animation whose palette changes per frame flickers): from a sample of
    # the frames side by side, 255 colors, the 256th kept for the transparent background
    n = len(frames)
    pick = frames if n <= 12 else [frames[int(i * (n - 1) / 11)] for i in range(12)]
    w, h = frames[0].size
    scale = min(1.0, 512.0 / max(w, h))
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    sheet = Image.new("RGB", (tw * len(pick), th))
    for k, f in enumerate(pick):
        sheet.paste(_flatten(f, matte).resize((tw, th)), (k * tw, 0))
    pal = sheet.quantize(colors=255 if transparent else 256, method=Image.Quantize.MEDIANCUT)
    out_frames = []
    for k, f in enumerate(frames):
        q = _flatten(f, matte).quantize(palette=pal, dither=Image.Dither.NONE)
        if transparent and f.mode == "RGBA":
            # a pixel at most half covered is the background (the GIF has no partial transparency)
            clear = f.getchannel("A").point(lambda a: 255 if a < 128 else 0)
            q.paste(255, mask=clear)
        out_frames.append(q)
        if progress and (k % 8 == 0 or k == n - 1):
            progress(0.2 + 0.7 * (k + 1) / n, "Writing the GIF")
    kw: Dict[str, Any] = {"save_all": True, "append_images": out_frames[1:], "duration": max(10, int(round(1000.0 / fps))),
                          "optimize": False, "disposal": 2 if transparent else 1}
    la = _loop_args(loop)
    if la is not None:
        kw["loop"] = la
    if transparent:
        kw["transparency"] = 255
    out_frames[0].save(str(out), format="GIF", **kw)


def _write_video(frames, out: Path, fmt, fps, loop, transparent, matte, progress):
    caps = capabilities()
    if not caps[fmt]:
        raise AnimError(caps["why"] or f"this ffmpeg has no {fmt.upper()} encoder; export a GIF")
    ff = caps["ffmpeg"]
    alpha = bool(transparent and fmt == "webm" and caps["alpha_video"])
    with tempfile.TemporaryDirectory(prefix="mace-anim-") as tmp:
        n = len(frames)
        for k, f in enumerate(frames):
            img = f if (alpha and f.mode == "RGBA") else _flatten(f, matte)
            # H.264 and VP9 need even sides
            w, h = img.size
            if w % 2 or h % 2:
                from PIL import Image
                pad = Image.new(img.mode, (w + w % 2, h + h % 2), (tuple(matte) + ((0,) if img.mode == "RGBA" else ())))
                pad.paste(img, (0, 0))
                img = pad
            img.save(os.path.join(tmp, f"f{k:05d}.png"))
            if progress and (k % 8 == 0 or k == n - 1):
                progress(0.1 + 0.4 * (k + 1) / n, "Preparing the frames")
        cmd = [ff, "-hide_banner", "-loglevel", "error", "-y"]
        plays = max(1, int(loop)) if int(loop) > 0 else 1
        if plays > 1:
            cmd += ["-stream_loop", str(plays - 1)]
        cmd += ["-framerate", f"{fps:g}", "-i", os.path.join(tmp, "f%05d.png")]
        if fmt == "webm":
            enc = caps["webm_encoder"]
            cmd += ["-c:v", enc, "-pix_fmt", "yuva420p" if alpha else "yuv420p", "-b:v", "0", "-crf", "30"]
            if alpha:
                cmd += ["-auto-alt-ref", "0"]
        else:
            enc = caps["mp4_encoder"]
            cmd += ["-c:v", enc, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
            if enc == "libx264":
                cmd += ["-crf", "20", "-preset", "medium"]
            else:
                cmd += ["-q:v", "3"]
        cmd += [str(out)]
        if progress:
            progress(0.55, f"Encoding with ffmpeg ({enc})")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 or not out.is_file():
            msg = (r.stderr or r.stdout or "").strip().splitlines()
            raise AnimError(f"ffmpeg could not write the {fmt.upper()}: {msg[-1] if msg else 'no message'}")


# ---------------------------------------------------------------- frames from files
def _symbols():
    from mace.gui.structures import _symbols as S
    return S()


def _frame(numbers, cart, cell=None, pbc=(False, False, False), origin=None, label: str = "") -> Dict[str, Any]:
    import numpy as np
    return {"numbers": [int(z) for z in numbers], "cart": np.asarray(cart, float).reshape(-1, 3),
            "cell": None if cell is None else np.asarray(cell, float), "pbc": [bool(p) for p in pbc],
            "origin": np.zeros(3) if origin is None else np.asarray(origin, float), "label": label}


def frames_from_path(path, mode: Optional[int] = None, frames: int = 24, amplitude: float = 1.0,
                     every: int = 1, source: str = "auto", amplitude_scale: str = "physical") -> Dict[str, Any]:
    """The frames of a file: {kind, title, frames: [{numbers, cart (Å), cell, pbc, origin, label}]}.

    A CRYSTAL output with normal modes: mode ``mode`` (default: the first real mode above 1 cm-1),
    ``frames`` per cycle, drawn as the GUI's mode view does: ``amplitude_scale`` "physical", CRYSTAL's
    printed zero-point amplitude x ``amplitude``; "normalized", the largest displacement 0.5 Å x
    ``amplitude``. A CRYSTAL optimization: its geometry steps. Anything else ASE reads with several frames
    (MD, NEB): every frame. ``every``: keep every n-th frame of a trajectory. ``source``: auto, freq,
    opt or ase."""
    p = Path(path)
    if not p.is_file():
        raise AnimError(f"{p}: no such file")
    every = max(1, int(every))
    suffix = p.suffix.lower()
    crystal = suffix in (".out", ".o", ".log") and source in ("auto", "freq", "opt")
    if crystal:
        from mace.gui import outindex
        try:
            idx = outindex.build(p)
        except Exception as exc:          # not a CRYSTAL output after all: ASE may read it
            if source != "auto":
                raise AnimError(f"{p.name}: not a CRYSTAL output ({exc})")
            idx = None
        if idx is not None and source == "freq" and not idx.freq:
            raise AnimError(f"{p.name} has no normal modes (FREQ with eigenvectors needed)")
        if idx is not None and idx.freq and source in ("freq", "auto"):
            return _freq_frames(p, idx, mode, frames, amplitude, amplitude_scale)
        if idx is not None and idx.n_steps:
            if source == "freq":
                raise AnimError(f"{p.name} has no normal modes (FREQ with eigenvectors needed)")
            out = []
            for k in range(0, idx.n_steps, every):
                f = idx.frame(k)
                s = f["structure"]
                e = f.get("energy_au")
                out.append(_frame(s["numbers"], s["cart"], s["cell"], s["pbc"], s.get("origin"),
                                  f"Step {k + 1} of {idx.n_steps} ({f['label']})" + (f" · E = {e:.8f} Ha" if e is not None else "")))
            if len(out) < 2:
                raise AnimError(f"{p.name} has one geometry: nothing to animate (an optimization prints one per step)")
            return {"kind": "opt", "title": f"{p.name}: optimization, {idx.n_steps} steps", "frames": out}
        if source in ("freq", "opt"):
            raise AnimError(f"{p.name} has no {'normal modes' if source == 'freq' else 'geometry steps'}")
    try:
        from ase.io import read as ase_read
    except ImportError:
        raise AnimError("reading MD and NEB trajectories needs ASE (pip install ase)")
    try:
        images = ase_read(str(p), index=":")
    except Exception as exc:
        raise AnimError(f"{p.name}: ASE cannot read it as a trajectory ({exc})")
    if not isinstance(images, list):
        images = [images]
    if len(images) < 2:
        raise AnimError(f"{p.name} holds one structure: nothing to animate")
    out = []
    n = len(images)
    for k in range(0, n, every):
        a = images[k]
        pbc = [bool(x) for x in a.get_pbc()]
        cell = a.get_cell().array if any(pbc) else None
        lab = f"Frame {k + 1} of {n}"
        try:
            e = a.get_potential_energy()
            lab += f" · E = {e:.4f} eV"
        except Exception:
            pass
        out.append(_frame(a.get_atomic_numbers(), a.get_positions(), cell, pbc, None, lab))
    return {"kind": "trajectory", "title": f"{p.name}: {n} frames", "frames": out}


def _freq_frames(p: Path, idx, mode: Optional[int], frames: int, amplitude: float,
                 amplitude_scale: str = "physical") -> Dict[str, Any]:
    import numpy as np
    from mace.gui import freq as gui_freq
    start, end = idx.freq
    if end - start > gui_freq.SECTION_MAX:
        raise AnimError("the FREQ section is too large to animate here")
    if not idx.n_steps:
        raise AnimError(f"{p.name} has normal modes but no geometry")
    geo = idx.frame(-1)["structure"]
    with tempfile.TemporaryDirectory(prefix="mace-freq-") as tmp:
        parsed = gui_freq.parse_job(None, path=str(p), start=start, end=end, tmpdir=tmp)
    if idx.dim == 0:
        parsed["rotation"] = np.eye(3)
    else:
        try:
            parsed["rotation"] = gui_freq.frame_rotation(idx._lines(idx.blocks[-1]), geo["cell"])
        except Exception:
            parsed["rotation"] = None
    modes = parsed["modes"]
    if not modes:
        raise AnimError(f"{p.name}: no normal modes")
    if mode is None:
        real = [m for m in modes if not m["imaginary"] and m["freq"] > 1.0]
        mode = (real or modes)[0]["mode"]
    m = next((x for x in modes if x["mode"] == int(mode)), None)
    if m is None:
        raise AnimError(f"no mode {mode}: modes run from {modes[0]['mode']} to {modes[-1]['mode']}")
    mp = gui_freq.mode_payload(parsed, int(mode))
    if mp["displacement"] is None:
        raise AnimError(f"mode {mode} has no eigenvector in {p.name}")
    disp = np.asarray(mp["displacement"], float)
    cart = np.asarray(geo["cart"], float)
    if len(disp) != len(cart):
        raise AnimError(f"the modes move {len(disp)} atoms, the geometry has {len(cart)}")
    if amplitude_scale not in AMPLITUDE_SCALES:
        raise AnimError(f"the amplitude scale is one of {', '.join(AMPLITUDE_SCALES)}")
    top = float(np.linalg.norm(disp, axis=1).max())
    if amplitude_scale == "physical":
        k0 = 1.0                        # the displacement as CRYSTAL prints it (Å)
    else:
        k0 = 0.5 / top if top > 1e-9 else 0.0
    nf = max(4, min(240, int(frames)))
    fr = "%.2f" % m["freq"]
    tag = f"Mode {m['mode']}: {'i' if m['imaginary'] else ''}{fr.lstrip('-')} cm⁻¹" + (f" ({m['irrep']})" if m.get("irrep") else "")
    out = []
    for f in range(nf):
        w = math.sin(2 * math.pi * f / nf) * float(amplitude) * k0
        out.append(_frame(geo["numbers"], cart + w * disp, geo["cell"], geo["pbc"], geo.get("origin"), tag))
    return {"kind": "freq", "title": f"{p.name}: {tag}", "frames": out, "mode": m}


# ---------------------------------------------------------------- the headless renderer
def _palette(name: Optional[str]) -> Dict[str, str]:
    from mace.utils import element_palettes as EP
    try:
        cols = EP.load_colors(name)
    except (ValueError, KeyError) as exc:
        raise AnimError(str(exc))
    if cols is None:
        cols = EP.resolve(EP.presets()["default"])
    return cols


def _rgb(hexc: str) -> Tuple[int, int, int]:
    h = hexc.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _lum(c) -> float:
    def f(x):
        x /= 255.0
        return x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(c[0]) + 0.7152 * f(c[1]) + 0.0722 * f(c[2])


def _ratio(a, b) -> float:
    x, y = _lum(a), _lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def gray_of(c) -> Tuple[int, int, int]:
    """The gray of the same lightness (the Grayscale print style)."""
    y = _lum(c)
    v = int(round(255 * (y * 12.92 if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055)))
    return v, v, v


def assign_patterns(symbols: Sequence[str], color_of: Callable[[str], Tuple[int, int, int]], mode: str) -> Dict[str, str]:
    """print_patterns.js assign(): "auto" patterns all but the most frequent element of each group of
    similar lightness (contrast < 1.35); "all" every element (most frequent first)."""
    out: Dict[str, str] = {}
    if mode not in ("auto", "all"):
        return out
    count: Dict[str, int] = {}
    for s in symbols:
        count[s] = count.get(s, 0) + 1
    els = sorted(count, key=lambda s: (-count[s], s))
    nxt = 0
    if mode == "all":
        for e in els:
            out[e] = PATTERNS[nxt % len(PATTERNS)]
            nxt += 1
        return out
    done = set()
    for e in els:
        if e in done:
            continue
        group = [x for x in els if x not in done and _ratio(color_of(x), color_of(e)) < SIMILAR]
        for k, x in enumerate(group):
            done.add(x)
            if k:
                out[x] = PATTERNS[nxt % len(PATTERNS)]
                nxt += 1
    return out


def _tile(pid: str, ink, px: int):
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w = max(1, px // 8)
    if pid in ("slash", "cross"):
        d.line([(0, px), (px, 0)], fill=ink, width=w)
        d.line([(-px // 2, px // 2), (px // 2, -px // 2)], fill=ink, width=w)
        d.line([(px // 2, px + px // 2), (px + px // 2, px // 2)], fill=ink, width=w)
    if pid in ("backslash", "cross"):
        d.line([(0, 0), (px, px)], fill=ink, width=w)
        d.line([(-px // 2, px // 2), (px // 2, px + px // 2)], fill=ink, width=w)
        d.line([(px // 2, -px // 2), (px + px // 2, px // 2)], fill=ink, width=w)
    if pid in ("horizontal", "grid"):
        d.line([(0, px // 2), (px, px // 2)], fill=ink, width=w)
    if pid in ("vertical", "grid"):
        d.line([(px // 2, 0), (px // 2, px)], fill=ink, width=w)
    if pid == "stipple":
        r = max(1, int(px * 0.09))
        for cx, cy in ((px * 0.25, px * 0.25), (px * 0.75, px * 0.7)):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ink)
    if pid == "dots":
        r = max(1, int(px * 0.2))
        d.ellipse([px / 2 - r, px / 2 - r, px / 2 + r, px / 2 + r], fill=ink)
    return img


def _view_rotation(view: str, frames) -> "Any":
    """Rows: screen right, up, toward the viewer, in model coordinates."""
    import numpy as np

    def basis(toward, up_hint):
        t = np.asarray(toward, float)
        t = t / np.linalg.norm(t)
        u = np.asarray(up_hint, float)
        u = u - np.dot(u, t) * t
        if np.linalg.norm(u) < 1e-6:
            u = np.array([0.0, 1.0, 0.0]) - t[1] * t
            if np.linalg.norm(u) < 1e-6:
                u = np.array([1.0, 0.0, 0.0]) - t[0] * t
        u = u / np.linalg.norm(u)
        r = np.cross(u, t)
        return np.vstack([r, u, t])
    f0 = frames[0]
    cell = f0["cell"]
    if view in ("x", "y", "z"):
        t = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[view]
        return basis(t, [0, 0, 1] if view != "z" else [0, 1, 0])
    if view in ("a", "b", "c") and cell is not None:
        k = "abc".index(view)
        t = cell[k]
        other = cell[(k + 2) % 3] if view != "c" else cell[1]
        return basis(t, other)
    if cell is not None and all(f0["pbc"]):
        # a crystal: the classic oblique look (c up, a toward the viewer's lower left)
        R = basis([0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
        rx, rz = math.radians(-65), math.radians(-25)
        Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
        Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        return (Rx @ Rz) @ R
    # a molecule, a slab or a chain: looking along the atoms' axis of least spread (a flat molecule face on)
    X = np.vstack([f["cart"] for f in frames])
    X = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return basis(vt[2], vt[1])


def _wrapped(frames):
    """The atoms of a periodic structure moved into the cell along its periodic axes, by the whole-cell shifts
    of the first frame for every frame (an atom never jumps across the cell between frames)."""
    import numpy as np
    f0 = frames[0]
    if f0["cell"] is None or not any(f0["pbc"]):
        return frames
    C = np.asarray(f0["cell"], float)
    try:
        inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return frames
    frac = (f0["cart"] - f0["origin"]) @ inv
    shift = -np.floor(frac + 1e-6) * np.asarray([1.0 if p else 0.0 for p in f0["pbc"]])
    out = []
    for f in frames:
        if f["cell"] is None or len(f["cart"]) != len(shift):
            out.append(f)
            continue
        out.append({**f, "cart": f["cart"] + shift @ np.asarray(f["cell"], float)})
    return out


def frame_bonds(cart, numbers, bond_rule: Optional[Dict[str, Any]] = None) -> List[Tuple[int, int]]:
    """The bonds between the drawn atoms of one frame (Cartesian, no periodic images), i < j, by the
    viewer's rule (structures.bond_pairs, geom/core.js cellListBonds): no Mg-Mg in MgO, a heavy
    cation's longer bonds to anions. ``bond_rule``: structures.bond_rule of the document's criteria."""
    import numpy as np
    from .structures import bond_pairs
    return [(i, j) for i, j, _ in bond_pairs(np.zeros((3, 3)), cart, numbers, (False, False, False), **(bond_rule or {}))]


def render_frames(src: Dict[str, Any], size: Tuple[int, int] = (800, 600), scale: float = 1.0, background: str = "light",
                  style: str = "ballstick", view: str = "auto", palette: Optional[str] = None, grayscale: bool = False,
                  patterns: str = "off", cell: bool = True, bonds: bool = True, label: bool = True,
                  supersample: int = 2, progress: Optional[Callable[[float, str], None]] = None,
                  bond_rule: Optional[Dict[str, Any]] = None) -> List[Any]:
    """Draw every frame of ``src`` (frames_from_path) as a PIL RGBA image of size x scale pixels, with one
    camera for the whole animation (fitted to every frame), so nothing jumps between frames. The bonds
    between the drawn atoms are the viewer's (structures.bond_pairs): ``bond_rule`` is
    structures.bond_rule of the document's criteria (None: the defaults)."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    if background not in BACKGROUNDS:
        raise AnimError(f"the background is one of {', '.join(BACKGROUNDS)}")
    if patterns not in PATTERN_MODES:
        raise AnimError(f"patterns is one of {', '.join(PATTERN_MODES)}")
    W, H = int(round(size[0] * scale)), int(round(size[1] * scale))
    if not (16 <= W <= MAX_SIDE and 16 <= H <= MAX_SIDE):
        raise AnimError(f"the picture is 16 to {MAX_SIDE} pixels a side ({W} x {H} asked)")
    frames = src["frames"]
    if len(frames) > MAX_FRAMES:
        raise AnimError(f"at most {MAX_FRAMES} frames (this one has {len(frames)}): keep every n-th (--every)")
    check_pixels(len(frames), W, H)
    try:
        from ase.data import covalent_radii, vdw_radii
    except ImportError:
        covalent_radii = vdw_radii = None
    sym = _symbols()
    cols = _palette(palette)
    fb = cols.get("_fallback") or "#909090"

    def color_of(s: str):
        c = _rgb(cols.get(s) or cols.get(s.capitalize()) or fb)
        return gray_of(c) if grayscale else c

    def radius(z: int) -> float:
        rc = float(covalent_radii[z]) if covalent_radii is not None and z < len(covalent_radii) else 0.8
        if style == "spacefill":
            rv = float(vdw_radii[z]) if vdw_radii is not None and z < len(vdw_radii) and not math.isnan(vdw_radii[z]) else 1.7
            return rv * 0.95
        return max(0.2, 0.5 * rc + 0.15)

    frames = _wrapped(frames)
    R = _view_rotation(view, frames)
    allx = np.vstack([f["cart"] for f in frames])
    corners = []
    for f in frames:
        if cell and f["cell"] is not None and all(f["pbc"]):
            C = f["cell"]
            corners += [f["origin"] + i * C[0] + j * C[1] + k * C[2] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    pts = np.vstack([allx] + ([np.asarray(corners)] if corners else []))
    rmax = max(radius(z) for z in set(frames[0]["numbers"]))
    P = pts @ R.T
    lo, hi = P[:, :2].min(axis=0) - rmax, P[:, :2].max(axis=0) + rmax
    ss = max(1, int(supersample))
    w2, h2 = W * ss, H * ss
    margin = 0.06
    span = np.maximum(hi - lo, 1e-6)
    s = min(w2 * (1 - 2 * margin) / span[0], h2 * (1 - 2 * margin - (0.06 if label else 0)) / span[1])
    mid = (lo + hi) / 2
    cx, cy = w2 / 2, h2 / 2 + (0.03 * h2 if label else 0)

    def to_screen(q):
        return cx + s * (q[0] - mid[0]), cy - s * (q[1] - mid[1])
    bg = BG[background]
    ink = INK[background]
    pmap = assign_patterns([sym[z] for z in frames[0]["numbers"]], color_of, patterns) if patterns != "off" else {}
    tiles: Dict[Tuple[str, Tuple[int, int, int]], Any] = {}
    font = None
    if label:
        try:
            font = ImageFont.load_default(size=max(10, int(13 * ss * scale)))
        except TypeError:
            font = ImageFont.load_default()
    out = []
    n = len(frames)
    for fi, f in enumerate(frames):
        img = Image.new("RGBA", (w2, h2), bg)
        d = ImageDraw.Draw(img)
        X = f["cart"]
        Q = X @ R.T
        zs = [int(z) for z in f["numbers"]]
        rad = [radius(z) for z in zs]
        items: List[Tuple[float, str, Any]] = []
        # the cell: its 12 edges, under everything
        if cell and f["cell"] is not None and all(f["pbc"]):
            C, o = f["cell"], f["origin"]
            cc = [o + i * C[0] + j * C[1] + k * C[2] for i in (0, 1) for j in (0, 1) for k in (0, 1)]
            for a in range(8):
                for bit in (4, 2, 1):
                    if a & bit:
                        continue
                    p1, p2 = to_screen(cc[a] @ R.T), to_screen(cc[a | bit] @ R.T)
                    d.line([p1, p2], fill=ink + (200,), width=max(1, int(1.2 * ss * scale)))
        if bonds and style != "spacefill" and covalent_radii is not None and len(X) <= 3000:
            for i, j in frame_bonds(X, zs, bond_rule):
                # under both of its atoms (each covers the stub it grows from)
                items.append((min(Q[i, 2], Q[j, 2]) - 1e-3, "b", (i, j)))
        for i in range(len(X)):
            items.append((Q[i, 2], "a", i))
        items.sort(key=lambda t: t[0])
        bw = max(1, int(round(2 * 0.08 * s)))
        for _, kind, what in items:
            if kind == "b":
                i, j = what
                p1, p2 = to_screen(Q[i]), to_screen(Q[j])
                m = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                for (pa, ci) in ((p1, i), (p2, j)):
                    c = tuple(int(v * 0.85) for v in color_of(sym[zs[ci]]))
                    d.line([pa, m], fill=c + (255,), width=bw)
                continue
            i = what
            x, y = to_screen(Q[i])
            r = rad[i] * s
            base = color_of(sym[zs[i]])
            dark = tuple(int(v * 0.45) for v in base)
            hl = tuple(int(v + (255 - v) * 0.55) for v in base)
            steps = max(8, min(28, int(r / 3)))
            for t in range(steps + 1):
                u = t / steps
                rr = r * (1 - 0.82 * u)
                ox, oy = -0.32 * r * u, -0.32 * r * u
                col = tuple(int(dark[k] + (base[k] - dark[k]) * min(1, u * 2.2) + (hl[k] - base[k]) * max(0.0, u - 0.6) / 0.4) for k in range(3))
                d.ellipse([x + ox - rr, y + oy - rr, x + ox + rr, y + oy + rr], fill=col + (255,))
            d.ellipse([x - r, y - r, x + r, y + r], outline=tuple(int(v * 0.35) for v in base) + (255,), width=max(1, ss))
            pid = pmap.get(sym[zs[i]])
            if pid and r >= 2:
                pink = (0, 0, 0, 158) if _lum(base) > 0.35 else (255, 255, 255, 199)
                key = (pid, pink)
                px = max(4, int(round(7 * ss * scale)))
                if key not in tiles:
                    tiles[key] = _tile(pid, pink, px)
                side = int(math.ceil(2 * r)) + 2
                layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                t0 = tiles[key]
                for ty in range(0, side, t0.size[1]):
                    for tx in range(0, side, t0.size[0]):
                        layer.paste(t0, (tx, ty))
                mask = Image.new("L", (side, side), 0)
                ImageDraw.Draw(mask).ellipse([1, 1, side - 2, side - 2], fill=255)
                from PIL import ImageChops
                layer.putalpha(ImageChops.multiply(layer.getchannel("A"), mask))
                left, top = int(round(x - side / 2)), int(round(y - side / 2))
                cl, ct = max(0, -left), max(0, -top)
                cr, cb = min(side, w2 - left), min(side, h2 - top)
                if cr > cl and cb > ct:
                    img.alpha_composite(layer.crop((cl, ct, cr, cb)), dest=(left + cl, top + ct))
        if label and f.get("label"):
            # the bitmap fonts Pillow ships have no superscripts
            text = f["label"].replace("cm⁻¹", "cm-1")
            d.text((int(10 * ss * scale), int(8 * ss * scale)), text, fill=ink + (255,), font=font)
        out.append(img.resize((W, H), Image.LANCZOS) if ss > 1 else img)
        if progress and (fi % 4 == 0 or fi == n - 1):
            progress(0.05 + 0.6 * (fi + 1) / n, f"Drawing frame {fi + 1} of {n}")
    return out


def check_pixels(n: int, w: int, h: int) -> None:
    """AnimError when ``n`` frames of ``w`` x ``h`` pixels are more than MAX_PIXELS (held at once to encode)."""
    if n * w * h > MAX_PIXELS:
        raise AnimError(f"{n} frames of {w} x {h} pixels are {n * w * h / 1e9:.2f} billion pixels, more than the "
                        f"{MAX_PIXELS / 1e9:g} billion an animation holds: keep every n-th frame (Every, --every) or "
                        "draw them smaller (Size, --size)")


def animate_file(path, out: Optional[Path] = None, fmt: str = "gif", fps: float = 20.0, loop: int = 0, mode: Optional[int] = None,
                 frames: int = 24, amplitude: float = 1.0, every: int = 1, source: str = "auto", size=(800, 600),
                 scale: float = 1.0, background: str = "light", view: str = "auto", style: str = "ballstick",
                 palette: Optional[str] = None, grayscale: bool = False, patterns: str = "off", cell: bool = True,
                 bonds: bool = True, label: bool = True, progress=None,
                 amplitude_scale: str = "physical", bond_rule: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`mace animate`: a file's frames drawn here and written as GIF/WebM/MP4."""
    src = frames_from_path(path, mode=mode, frames=frames, amplitude=amplitude, every=every, source=source,
                           amplitude_scale=amplitude_scale)
    imgs = render_frames(src, size=size, scale=scale, background=background, style=style, view=view, palette=palette,
                         grayscale=grayscale, patterns=patterns, cell=cell, bonds=bonds, label=label, progress=progress,
                         bond_rule=bond_rule)
    p = Path(path)
    if out is None:
        tag = f"_mode{src['mode']['mode']}" if src["kind"] == "freq" else f"_{src['kind']}"
        out = p.with_name(p.stem + tag + "." + fmt)
    res = encode(imgs, Path(out), fmt=fmt, fps=fps, loop=loop, transparent=background == "transparent",
                 matte=BG["light"][:3] if background == "transparent" else BG[background][:3], progress=progress)
    res.update(kind=src["kind"], title=src["title"])
    return res


# ---------------------------------------------------------------- the GUI's routes
_SESSIONS: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _root(st) -> Path:
    base = Path(getattr(st, "cache", None) or tempfile.gettempdir()) / "anim"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _prune(st) -> None:
    now = time.time()
    with _LOCK:
        old = [k for k, v in _SESSIONS.items() if now - v["t"] > SESSION_TTL]
        for k in old:
            shutil.rmtree(_SESSIONS.pop(k)["dir"], ignore_errors=True)


def _session(st, token: str) -> Dict[str, Any]:
    with _LOCK:
        s = _SESSIONS.get(str(token))
    if not s:
        raise AnimError("this capture was let go of (start the export again)")
    return s


def _decode_frame(data: str):
    import base64
    from PIL import Image
    if not isinstance(data, str) or "," not in data or not data.startswith("data:image/"):
        raise AnimError("a frame is a data: URL of a PNG or WebP image")
    head, b64 = data.split(",", 1)
    if not (head.startswith("data:image/png") or head.startswith("data:image/webp")) or ";base64" not in head:
        raise AnimError("a frame is a PNG or WebP image")
    raw = base64.b64decode(b64, validate=False)
    img = Image.open(io.BytesIO(raw))
    img.load()
    if max(img.size) > MAX_SIDE:
        raise AnimError(f"a frame is at most {MAX_SIDE} pixels a side")
    return img, raw


def get_route(handler, route: str, query):
    """GET /api/anim/caps (formats this machine writes) and /api/anim/result?key= (the file)."""
    st = handler.state
    if route == "anim/caps":
        c = capabilities()
        return handler._json({k: c[k] for k in ("gif", "webm", "mp4", "alpha_video", "why", "webm_encoder", "mp4_encoder")})
    if route == "anim/result":
        s = _session(st, handler._q(query, "key", ""))
        out = s.get("out")
        if not out or not Path(out).is_file():
            raise AnimError("the animation is not written yet")
        data = Path(out).read_bytes()
        ctype = {"gif": "image/gif", "webm": "video/webm", "mp4": "video/mp4"}[s["fmt"]]
        name = s.get("name") or Path(out).name
        if handler._q(query, "keep", "") != "1":
            with _LOCK:
                _SESSIONS.pop(s["token"], None)
            shutil.rmtree(s["dir"], ignore_errors=True)
        return handler._send(200, data, ctype, {"Cache-Control": "no-store",
                                                "Content-Disposition": f'attachment; filename="{name}"'})
    raise FileNotFoundError(route)


def post_route(handler, route: str, body: Dict[str, Any]):
    """POST /api/anim/start → {key}; anim/frame {key, index, data} (one captured frame); anim/encode {key,
    format, fps, loop, transparent, matte, name} → a task whose result names the file; anim/render {path, ...}:
    a file's frames drawn here (as `mace animate`), then encoded; anim/cancel {key}."""
    from . import tasks as gui_tasks
    st = handler.state
    if route == "anim/start":
        _prune(st)
        token = uuid.uuid4().hex
        d = _root(st) / token
        d.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            _SESSIONS[token] = {"token": token, "dir": str(d), "t": time.time(), "frames": {}, "fmt": "gif"}
        return handler._json({"key": token, "caps": {k: v for k, v in capabilities().items() if k in ("gif", "webm", "mp4", "alpha_video", "why")}})
    if route == "anim/cancel":
        with _LOCK:
            s = _SESSIONS.pop(str(body.get("key", "")), None)
        if s:
            shutil.rmtree(s["dir"], ignore_errors=True)
        return handler._json({"ok": True})
    if route == "anim/frame":
        s = _session(st, body.get("key", ""))
        try:
            k = int(body.get("index"))
        except (TypeError, ValueError):
            raise AnimError("index is the frame's number")
        if not 0 <= k < MAX_FRAMES:
            raise AnimError(f"at most {MAX_FRAMES} frames")
        img, raw = _decode_frame(body.get("data"))
        with _LOCK:
            px = {**s.get("px", {}), k: img.size[0] * img.size[1]}
        if sum(px.values()) > MAX_PIXELS:
            raise AnimError(f"the captured frames are more than the {MAX_PIXELS / 1e9:g} billion pixels an animation "
                            "holds: fewer frames (a larger step) or a smaller size")
        f = Path(s["dir"]) / f"f{k:05d}.{img.format.lower() if img.format else 'png'}"
        f.write_bytes(raw)
        with _LOCK:
            s["frames"][k] = str(f)
            s["px"] = px
            s["t"] = time.time()
        return handler._json({"ok": True, "n": len(s["frames"])})
    if route == "anim/encode":
        s = _session(st, body.get("key", ""))
        fmt = str(body.get("format", "gif")).lower()
        if fmt not in FORMATS:
            raise AnimError(f"the format is one of {', '.join(FORMATS)}")
        fps, loop = float(body.get("fps", 20)), int(body.get("loop", 0))
        transparent = bool(body.get("transparent", False))
        matte = _parse_hex(body.get("matte")) or (255, 255, 255)
        name = _safe_name(body.get("name"), fmt)
        files = [s["frames"][k] for k in sorted(s["frames"])]
        if not files:
            raise AnimError("no frames were captured")

        def job(ctx):
            from PIL import Image
            imgs = []
            for k, fpath in enumerate(files):
                ctx.check()
                im = Image.open(fpath)
                im.load()
                imgs.append(im.convert("RGBA"))
                if k % 8 == 0:
                    ctx.progress(0.2 * (k + 1) / len(files), "Reading the frames")
            out = Path(s["dir"]) / name
            r = encode(imgs, out, fmt=fmt, fps=fps, loop=loop, transparent=transparent, matte=matte,
                       progress=lambda x, m: ctx.progress(x, m))
            with _LOCK:
                s.update(out=str(out), fmt=fmt, name=name, t=time.time())
            return {**r, "key": s["token"], "name": name}
        t = st.tasks.submit("anim", f"Writing {name}", "", {}, client=handler.headers.get("X-Mace-Client"), inline=job)
        return gui_tasks.answer(handler, st.tasks, t, 60)
    if route == "anim/render":
        p = st.resolve(body.get("path"))
        if not p.is_file():
            raise FileNotFoundError(st.rel(p))
        fmt = str(body.get("format", "gif")).lower()
        name = _safe_name(body.get("name") or f"{p.stem}.{fmt}", fmt)
        opts = _render_opts(body)
        _prune(st)
        token = uuid.uuid4().hex
        d = _root(st) / token
        d.mkdir(parents=True, exist_ok=True)
        s = {"token": token, "dir": str(d), "t": time.time(), "frames": {}, "fmt": fmt}
        with _LOCK:
            _SESSIONS[token] = s

        def job(ctx):
            out = d / name
            r = animate_file(p, out=out, fmt=fmt, progress=lambda x, m: (ctx.check(), ctx.progress(x, m)), **opts)
            with _LOCK:
                s.update(out=str(out), fmt=fmt, name=name, t=time.time())
            return {**r, "key": token, "name": name}
        t = st.tasks.submit("anim", f"Animating {p.name}", "", {}, client=handler.headers.get("X-Mace-Client"), inline=job)
        return gui_tasks.answer(handler, st.tasks, t, 60)
    raise FileNotFoundError(route)


def _parse_hex(v) -> Optional[Tuple[int, int, int]]:
    if isinstance(v, str) and len(v) == 7 and v.startswith("#"):
        try:
            return _rgb(v)
        except ValueError:
            return None
    return None


def _safe_name(name, fmt: str) -> str:
    n = Path(str(name or f"animation.{fmt}")).name
    n = "".join(ch if ch.isalnum() or ch in "._-+()" else "_" for ch in n).strip("._") or "animation"
    if not n.lower().endswith("." + fmt):
        n = n.rsplit(".", 1)[0] if n.lower().rsplit(".", 1)[-1] in FORMATS else n
        n += "." + fmt
    return n[:160]


def _render_opts(body: Dict[str, Any]) -> Dict[str, Any]:
    from . import structures
    def num(k, d, lo, hi, cast=float):
        try:
            x = cast(body.get(k, d))
        except (TypeError, ValueError):
            raise AnimError(f"{k} is a number")
        if not (lo <= x <= hi):
            raise AnimError(f"{k} runs from {lo} to {hi}")
        return x
    bgv = str(body.get("background", "light"))
    if bgv not in BACKGROUNDS:
        raise AnimError(f"the background is one of {', '.join(BACKGROUNDS)}")
    pat = str(body.get("patterns", "off"))
    if pat not in PATTERN_MODES:
        raise AnimError(f"patterns is one of {', '.join(PATTERN_MODES)}")
    mode = body.get("mode")
    # the FREQ view's scale: Physical runs x1 to x100, the 0.5 Å fit 0.2 to 4
    ascale = str(body.get("amplitude_scale", "physical"))
    if ascale not in AMPLITUDE_SCALES:
        raise AnimError(f"amplitude_scale is one of {', '.join(AMPLITUDE_SCALES)}")
    out = {"fps": num("fps", 20, 0.5, MAX_FPS), "loop": num("loop", 0, 0, 1000, int),
           "mode": None if mode in (None, "") else num("mode", 1, 1, 100000, int),
           "frames": num("frames", 24, 4, 240, int), "amplitude_scale": ascale,
           "amplitude": num("amplitude", 1.0, 0.05, 100.0 if ascale == "physical" else 10.0),
           "every": num("every", 1, 1, 1000, int), "size": (num("width", 800, 16, MAX_SIDE, int), num("height", 600, 16, MAX_SIDE, int)),
           "scale": num("scale", 1.0, 0.25, 4.0), "background": bgv, "view": str(body.get("view", "auto")),
           "patterns": pat, "grayscale": bool(body.get("grayscale", False)), "label": bool(body.get("label", True)),
           # the document's bond criteria (Build › Calculate Bonds…): the bonds the viewer draws
           "bond_rule": structures.bond_rule(body.get("bonds")) if isinstance(body.get("bonds"), dict) else None}
    # a normal mode's frame count is known now (a trajectory's once it is read: render_frames checks it then)
    w, h = (int(round(x * out["scale"])) for x in out["size"])
    if out["mode"] is not None or str(body.get("source", "")) == "freq":
        check_pixels(out["frames"], w, h)
    return out


# ---------------------------------------------------------------- `mace animate`
def build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="mace animate",
        description=("Animate a CRYSTAL normal mode (FREQ output), an optimization trajectory (OPT output), or an MD run "
                     "or NEB path (any multi-frame file ASE reads: .xyz, .extxyz, .traj) as GIF, WebM or MP4. "
                     "The frames are drawn headless (Pillow); WebM and MP4 need ffmpeg on PATH. "
                     "The GUI's File > Export Animation... captures its own 3D view instead."),
        epilog=("examples:\n"
                "  mace animate run_freq.out --mode 7                 # mode 7 as run_freq_mode7.gif\n"
                "  mace animate run_freq.out --mode 7 --format mp4 --fps 30\n"
                "  mace animate run_freq.out --mode 7 --amplitude 10  # the physical amplitude x10\n"
                "  mace animate opt.out --every 2 --size 1280x720 --background white\n"
                "  mace animate md.extxyz --format webm --background transparent\n"
                "  mace animate neb.xyz --grayscale --patterns auto   # black-and-white print\n"
                "  mace animate --check                               # which formats this machine writes"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="CRYSTAL .out (FREQ or OPT) or a multi-frame structure file")
    p.add_argument("-o", "--output", help="output file (default: beside the input, <stem>_mode<N>.<format>)")
    p.add_argument("--format", choices=FORMATS, help="gif (default; or from the -o suffix), webm or mp4")
    p.add_argument("--source", choices=("auto", "freq", "opt", "ase"), default="auto",
                   help="what to animate in a CRYSTAL output (auto: its normal modes if it has them, else its steps)")
    p.add_argument("--mode", type=int, help="the normal mode (default: the first real one above 1 cm-1)")
    p.add_argument("--list-modes", action="store_true", help="list the output's normal modes and stop")
    p.add_argument("--frames", type=int, default=24, help="frames per vibration cycle (default 24, as the GUI)")
    p.add_argument("--amplitude", type=float, default=1.0,
                   help="FREQ: multiplies the drawn displacement (default 1: the physical zero-point amplitude "
                        "CRYSTAL prints; with --amplitude-scale normalized, the largest atom moves 0.5 A x this)")
    p.add_argument("--amplitude-scale", choices=AMPLITUDE_SCALES, default="physical",
                   help="FREQ: physical (default, as the GUI) or normalized (the largest atom 0.5 A x --amplitude, "
                        "the GUI's Fit 0.5 A and the drawing before 1.1.6)")
    p.add_argument("--every", type=int, default=1, help="keep every n-th frame of a trajectory (default 1)")
    p.add_argument("--fps", type=float, default=20.0, help="frames per second (default 20)")
    p.add_argument("--loop", type=int, default=0, help="plays: 0 = forever (default), 1 = once, n = n times")
    p.add_argument("--size", default="800x600", help="width x height in pixels (default 800x600)")
    p.add_argument("--scale", type=float, default=1.0, help="multiply the size (default 1; 2 for a sharper picture)")
    p.add_argument("--background", choices=BACKGROUNDS, default="light",
                   help="light / dark (MACE's theme surfaces), white, black, or transparent (GIF, WebM)")
    p.add_argument("--view", default="auto", help="auto, a, b, c (along a lattice vector) or x, y, z")
    p.add_argument("--style", choices=("ballstick", "spacefill"), default="ballstick")
    p.add_argument("--palette", help="element colors (a palette of View > Element Colors; default: your saved default)")
    p.add_argument("--grayscale", action="store_true", help="atoms in grays of the same lightness (for print)")
    p.add_argument("--patterns", choices=PATTERN_MODES, default=None,
                   help="print patterns on the atoms: off, auto (similar grays only), all (default: auto with --grayscale, else off)")
    p.add_argument("--no-cell", action="store_true", help="leave out the unit cell")
    p.add_argument("--no-bonds", action="store_true", help="leave out the bonds")
    p.add_argument("--no-label", action="store_true", help="no caption (mode, step, energy)")
    p.add_argument("--check", action="store_true", help="say which formats this machine can write, then stop")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    import sys
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.check:
        c = capabilities()
        print("GIF:  yes (Pillow)")
        print(f"WebM: {'yes (' + str(c['webm_encoder']) + (', transparent' if c['alpha_video'] else '') + ')' if c['webm'] else 'no'}")
        print(f"MP4:  {'yes (' + str(c['mp4_encoder']) + ')' if c['mp4'] else 'no'}")
        if c["why"]:
            print(c["why"])
        return 0
    if not args.file:
        build_parser().print_usage(sys.stderr)
        print("mace animate: give a file (or --check)", file=sys.stderr)
        return 2
    try:
        if args.list_modes:
            from mace.gui import outindex
            from mace.gui import freq as gui_freq
            p = Path(args.file)
            idx = outindex.build(p)
            if not idx.freq:
                raise AnimError(f"{p.name} has no normal modes")
            start, end = idx.freq
            with tempfile.TemporaryDirectory(prefix="mace-freq-") as tmp:
                parsed = gui_freq.parse_job(None, path=str(p), start=start, end=end, tmpdir=tmp)
            print(f"{'Mode':>6} {'cm-1':>12} {'irrep':>8}  IR  Raman")
            for m in parsed["modes"]:
                fq = f"{'i' if m['imaginary'] else ''}{abs(m['freq']):.2f}"
                print(f"{m['mode']:>6} {fq:>12} {m.get('irrep') or '':>8}  {'yes' if m['ir_active'] else 'no ':>3} {'yes' if m['raman_active'] else 'no'}")
            return 0
        fmt = args.format
        if not fmt and args.output and Path(args.output).suffix.lower().lstrip(".") in FORMATS:
            fmt = Path(args.output).suffix.lower().lstrip(".")
        fmt = fmt or "gif"
        try:
            w, h = (int(x) for x in str(args.size).lower().replace("×", "x").split("x"))
        except ValueError:
            raise AnimError("--size is WIDTHxHEIGHT, e.g. 800x600")
        patterns = args.patterns or ("auto" if args.grayscale else "off")
        out = Path(args.output) if args.output else None
        if out is not None and out.suffix.lower().lstrip(".") != fmt:
            out = out.with_suffix("." + fmt)
        last = [0.0]

        def progress(x, msg):
            if x - last[0] >= 0.2 or x >= 0.999:
                last[0] = x
                print(f"  {int(100 * x):3d}%  {msg}", file=sys.stderr)
        r = animate_file(args.file, out=out, fmt=fmt, fps=args.fps, loop=args.loop, mode=args.mode, frames=args.frames,
                         amplitude=args.amplitude, every=args.every, source=args.source, size=(w, h), scale=args.scale,
                         background=args.background, view=args.view, style=args.style, palette=args.palette,
                         grayscale=args.grayscale, patterns=patterns, cell=not args.no_cell, bonds=not args.no_bonds,
                         label=not args.no_label, progress=progress, amplitude_scale=args.amplitude_scale)
    except AnimError as exc:
        print(f"mace animate: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {r['path']} ({r['frames']} frames, {r['size'][0]} x {r['size'][1]}, {r['fps']:g} fps, "
          f"{r['bytes'] / 1e6:.2f} MB): {r['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
