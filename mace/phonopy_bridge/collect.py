"""Check the displaced-supercell outputs and turn their forces into phonons."""

from __future__ import annotations

import contextlib
import io
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import parent as P
from .writer import DISP_YAML, read_manifest

GEOMETRY_TOLERANCE = 1e-4          # Angstrom
IMAGINARY_THRESHOLD = -0.1         # THz
GAMMA_RADIUS = 1e-3                # |q| (fractional) treated as Gamma
CM1_PER_THZ = 33.35641

_FORCE_HEADER = "CARTESIAN FORCES IN HARTREE/BOHR"
_FORCE_ROW = re.compile(r"^\s*\d+\s+\d+\s+(-?[\d.]+E?[-+]?\d*\s+){2}-?[\d.]+E?[-+]?\d*\s*$")
_CART_ROW = re.compile(r"^\s*(\d+)\s+(\d+)\s+([A-Za-z]{1,2})\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


@dataclass
class OutputCheck:
    name: str
    problems: List[str] = field(default_factory=list)


def _float(s: str) -> float:
    return float(s.replace("D", "E"))


def printed_geometry(out_text: str) -> Optional[Tuple[np.ndarray, List[int], np.ndarray]]:
    """(lattice 3x3, atomic numbers, cartesian positions) from the FIRST
    'DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)' and
    'CARTESIAN COORDINATES - PRIMITIVE CELL' blocks of a CRYSTAL output."""
    lines = out_text.splitlines()
    lat = None
    for i, ln in enumerate(lines):
        if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)" in ln:
            try:
                lat = np.array([[_float(x) for x in lines[i + k].split()[:3]] for k in (2, 3, 4)])
            except (ValueError, IndexError):
                return None
            break
    if lat is None:
        return None
    for i, ln in enumerate(lines):
        if "CARTESIAN COORDINATES - PRIMITIVE CELL" in ln:
            z, pos = [], []
            j = i + 1
            while j < len(lines) and (lines[j].strip().startswith("*") or not lines[j].strip()):
                j += 1
            while j < len(lines):
                m = _CART_ROW.match(lines[j])
                if not m:
                    break
                try:
                    pos.append([_float(m.group(k)) for k in (4, 5, 6)])
                except ValueError:
                    break
                z.append(int(m.group(2)))
                j += 1
            if pos:
                return lat, z, np.array(pos)
            return None
    return None


def force_rows(out_text: str) -> int:
    """Number of atom rows in the last CARTESIAN FORCES block (0 when none)."""
    lines = out_text.splitlines()
    idx = [i for i, ln in enumerate(lines) if _FORCE_HEADER in ln]
    if not idx:
        return 0
    j = idx[-1] + 2                 # header, then the "ATOM  X  Y  Z" hook line
    n = 0
    while j < len(lines) and _FORCE_ROW.match(lines[j]):
        n += 1
        j += 1
    return n


def geometry_mismatch(out_text: str, cell, tol: float = GEOMETRY_TOLERANCE) -> Optional[str]:
    """Why the output's printed geometry is not phonopy's displaced cell, or None."""
    geo = printed_geometry(out_text)
    if geo is None:
        return "no printed geometry (lattice vectors / cartesian coordinates) found"
    lat, z, pos = geo
    ref_lat = np.array(cell.cell)
    if lat.shape != (3, 3) or np.abs(lat - ref_lat).max() > tol:
        return (f"lattice vectors differ from the displaced cell by "
                f"{np.abs(lat - ref_lat).max():.2e} A")
    if len(z) != len(cell.numbers):
        return f"{len(z)} atoms printed, the displaced cell has {len(cell.numbers)}"
    if [a % 100 for a in z] != [int(n) for n in cell.numbers]:
        return "element order differs from the displaced cell"
    d = pos - np.array(cell.positions)
    frac = d @ np.linalg.inv(ref_lat)
    frac -= np.round(frac)
    dist = np.linalg.norm(frac @ ref_lat, axis=1)
    worst = int(np.argmax(dist))
    if dist[worst] > tol:
        return (f"atom {worst + 1} is {dist[worst]:.2e} A away from its position in the "
                "displaced cell")
    return None


def check_outputs(outdir: Path, names: Sequence[str], cells) -> List[OutputCheck]:
    results = []
    n_super = len(cells[0].numbers) if len(cells) else 0
    for name, cell in zip(names, cells):
        r = OutputCheck(name)
        out = outdir / f"{name}.out"
        if not out.is_file():
            r.problems.append("no output (not run yet?)")
            results.append(r)
            continue
        text = out.read_text(errors="replace")
        if not P.normal_termination(text):
            r.problems.append("did not terminate normally")
        if "SCF ENDED - CONVERGENCE ON ENERGY" not in text:
            r.problems.append("SCF did not converge (no 'SCF ENDED - CONVERGENCE ON ENERGY')")
        rows = force_rows(text)
        if rows != n_super:
            r.problems.append(f"force block has {rows} rows, expected {n_super}")
        why = geometry_mismatch(text, cell)
        if why:
            r.problems.append(
                f"geometry is not this displacement: {why}. A stale job script that does "
                "not stage the .ext? Resubmit with `mace submit --overwrite-sh`")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Band paths
# ---------------------------------------------------------------------------

def _num(tok: str) -> float:
    return float(Fraction(tok))


def parse_band(spec: str) -> List[List[List[float]]]:
    """phonopy BAND syntax: points as 3 numbers (fractions allowed), commas
    separate disconnected paths."""
    paths = []
    for chunk in spec.split(","):
        vals = [_num(t) for t in chunk.split()]
        if len(vals) % 3 or len(vals) < 6:
            raise ValueError(f"band path segment {chunk.strip()!r} needs at least 2 points of 3 numbers")
        paths.append([vals[i:i + 3] for i in range(0, len(vals), 3)])
    return paths


def auto_band(cell) -> Tuple[List[List[List[float]]], List[str]]:
    """SeeK-path's path for the phonopy unit cell (= CRYSTAL's primitive cell),
    with its points rewritten in that cell's reciprocal basis."""
    import sys
    cd3 = str(Path(__file__).resolve().parents[2] / "Crystal_d3")
    if cd3 not in sys.path:
        sys.path.insert(0, cd3)
    with contextlib.redirect_stdout(io.StringIO()):
        import seekpath  # noqa: F401  (ImportError reaches the caller)
        import seekpath_interface as spi
        res = spi.get_seekpath_bandpath(np.array(cell.cell), np.array(cell.scaled_positions),
                                        list(cell.numbers))
        res = spi.to_crystal_basis(res, np.array(cell.cell), np.array(cell.scaled_positions),
                                   list(cell.numbers))
    coords = res["point_coords"]
    paths: List[List[List[float]]] = []
    names: List[List[str]] = []
    for a, b in res["path"]:
        if paths and names[-1][-1] == a:
            paths[-1].append(list(coords[b]))
            names[-1].append(b)
        else:
            paths.append([list(coords[a]), list(coords[b])])
            names.append([a, b])
    # phonopy takes one label per path end point, all paths concatenated.
    labels = [("$\\Gamma$" if s == "GAMMA" else s) for seg in names for s in seg]
    return paths, labels


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

@dataclass
class CollectResult:
    checks: List[OutputCheck]
    files: List[str] = field(default_factory=list)
    drifts: List[str] = field(default_factory=list)
    min_frequency: Optional[float] = None
    imaginary: List[Tuple[List[float], float]] = field(default_factory=list)


def load_set(outdir: Path):
    import phonopy

    with contextlib.redirect_stdout(io.StringIO()):
        phonon = phonopy.load(str(outdir / DISP_YAML), produce_fc=False, log_level=0)
    return phonon


def collect(outdir: Path, check_only: bool = False, band: Optional[str] = "auto",
            band_labels: Optional[str] = None, band_points: int = 51,
            mesh="100.0", tmin: float = 0.0, tmax: float = 1000.0, tstep: float = 10.0,
            plot: bool = False, log=print) -> CollectResult:
    from phonopy.file_IO import write_FORCE_SETS
    from phonopy.interface.crystal import parse_set_of_forces

    outdir = Path(outdir)
    man = read_manifest(outdir)
    names = man["jobs"]
    phonon = load_set(outdir)
    cells = phonon.supercells_with_displacements
    if len(cells) != len(names):
        raise ValueError(f"{DISP_YAML} has {len(cells)} displacements but the manifest lists "
                         f"{len(names)} decks")
    checks = check_outputs(outdir, names, cells)
    result = CollectResult(checks=checks)
    if any(c.problems for c in checks) or check_only:
        return result

    n_super = len(cells[0].numbers)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        forces = parse_set_of_forces(n_super, [str(outdir / f"{n}.out") for n in names])
    result.drifts = [ln.strip() for ln in buf.getvalue().splitlines() if "drift" in ln.lower()]
    if len(forces) != len(names):
        raise ValueError("phonopy could not read forces from every output:\n" + buf.getvalue())
    phonon.forces = np.array(forces)
    with contextlib.redirect_stdout(io.StringIO()):
        phonon.produce_force_constants()
        # As phonopy's own command line and phonopy.load do: impose the
        # space-group, translational and index symmetries that finite
        # differences only satisfy approximately (symfc projector).
        phonon.symmetrize_force_constants(use_symfc_projector=True)

    with contextlib.chdir(outdir):
        phonon.save("phonopy_params.yaml")
        write_FORCE_SETS(phonon.dataset, filename="FORCE_SETS")
        result.files += ["phonopy_params.yaml", "FORCE_SETS"]

        prim = phonon.primitive
        if band:
            if band == "auto":
                paths, labels = auto_band(prim)
            else:
                paths = parse_band(band)
                labels = band_labels.split() if band_labels else None
            from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
            qpoints, connections = get_band_qpoints_and_path_connections(paths, npoints=band_points)
            phonon.run_band_structure(qpoints, path_connections=connections, labels=labels)
            phonon.write_yaml_band_structure(filename="band.yaml")
            result.files.append("band.yaml")
            bs = phonon.band_structure
            for qs, fs in zip(bs.qpoints, bs.frequencies):
                for q, f in zip(qs, fs):
                    if np.linalg.norm(q) > GAMMA_RADIUS and f[0] < IMAGINARY_THRESHOLD:
                        result.imaginary.append((list(map(float, q)), float(f[0])))

        mesh_arg = _mesh_arg(mesh)
        phonon.run_mesh(mesh_arg)
        phonon.write_yaml_mesh()
        result.files.append("mesh.yaml")
        mesh_res = phonon.mesh
        mins = []
        for q, f in zip(mesh_res.qpoints, mesh_res.frequencies):
            mins.append(float(f[0]))
            if np.linalg.norm(q) > GAMMA_RADIUS and f[0] < IMAGINARY_THRESHOLD:
                result.imaginary.append((list(map(float, q)), float(f[0])))
        result.min_frequency = min(mins) if mins else None
        phonon.run_total_dos()
        phonon.write_total_dos(filename="total_dos.dat")
        phonon.run_thermal_properties(t_min=tmin, t_max=tmax, t_step=tstep)
        phonon.write_yaml_thermal_properties(filename="thermal_properties.yaml")
        result.files += ["total_dos.dat", "thermal_properties.yaml"]

        if plot:
            import matplotlib
            matplotlib.use("Agg")
            if band:
                plt = phonon.plot_band_structure_and_dos()
            else:
                plt = phonon.plot_total_dos()
            plt.savefig("band_dos.pdf")
            plt.close("all")
            result.files.append("band_dos.pdf")
    return result


def _mesh_arg(mesh):
    if isinstance(mesh, (int, float)):
        return float(mesh)
    toks = str(mesh).split()
    if len(toks) == 1:
        return float(toks[0])
    if len(toks) == 3:
        return [int(t) for t in toks]
    raise ValueError("--mesh takes one length or three integers")
