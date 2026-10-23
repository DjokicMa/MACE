"""Reference energies: quantities that combine the run with other runs.

* formation energy per atom, against elemental reference runs;
* cohesive energy per atom, against isolated-atom runs;
* surface energy of a slab, (E_slab - n E_bulk) / 2A, against a bulk run;
* adsorption energy, E(complex) - E(surface) - E(adsorbate), per adsorbate;
* reaction energy, Σ products - Σ reactants, with stoichiometric coefficients.

Each as ΔE (from the energies with D3 and gCP), and as ΔH and ΔG when every
run has FREQ thermodynamics at one temperature (its own, or a FREQ run of
the same material in the database); else ΔE only, and the property's JSON
says why. Runs computed with another functional, basis set or CRYSTAL
version are refused (mace.properties.reference.check says what refuses and
what only warns). The references are the runs the user gives
(``--reference ROLE=FILE|CALC_ID``, the Properties tab's Add Reference…),
the run's saved choice, or a lookup in the database (same functional and
basis set). BSSE (counterpoise) and charged-defect corrections are not
applied. docs/properties.md, "Reference energies".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from mace.properties import reference as R
from mace.properties.registry import Output, Reference, Skip, derived

TYPES = R.REF_TYPES
GROUP = "Reference energies"
_FORMS = (("energy", "Δ<i>E</i>", "energy"), ("enthalpy", "Δ<i>H</i>", "enthalpy"),
          ("gibbs_energy", "Δ<i>G</i>", "Gibbs energy"))


def _outputs(prefix: str, sub: str, units, what: str, detail: str):
    """ΔE, ΔH and ΔG outputs in each unit (``units``: (suffix, unit)), and the JSON details. The
    first unit's key is the one the Properties tab shows (the others are ``same_as`` it)."""
    out = []
    for form, sym, word in _FORMS:
        sym = sym.replace("<i>", "").replace("</i>", "")
        for i, (suffix, unit) in enumerate(units):
            out.append(Output(f"{prefix}_{form}_{suffix}", unit, label=f"{sym}<sub>{sub}</sub>", precision=6,
                              group=GROUP, category="thermodynamics" if form != "energy" else "energy",
                              same_as=f"{prefix}_{form}_{units[0][0]}" if i else "",
                              description=f"{what}: the {word} form" + (
                                  "" if form == "energy" else " (FREQ thermodynamics of every run, at its temperature)")))
    out.append(Output(f"{prefix}_energy", "JSON", "json", label=f"{what[:1].upper() + what[1:]} details", group=GROUP,
                      description=detail))
    return tuple(out)


def _detail(formula_text: str, parts: List[R.Part], warnings: List[str], extra: Dict[str, Any],
            values: Dict[str, Any], thermo: Dict[str, Any]) -> Dict[str, Any]:
    d = {"formula": formula_text, "runs": [p.info() for p in parts], "warnings": warnings,
         "not_included": "BSSE (counterpoise) and charged-defect corrections"}
    d.update(extra)
    d.update(thermo)
    d["values"] = values
    return R.json_safe(d)


def _store(prefix: str, forms: Dict[str, Dict[str, float]], detail: Dict[str, Any]) -> Dict[str, Any]:
    vals = {f"{prefix}_{form}_{suffix}": v for form, d in forms.items() for suffix, v in d.items()}
    detail["values"] = dict(vals)
    vals[f"{prefix}_energy"] = detail
    return vals


def _parts(ctx, role: str) -> List[R.Part]:
    return [R.Part.of_ref(r, role, ctx.db) for r in ctx.references.get(role) or []]


def _need_comp(parts: List[R.Part]) -> None:
    unknown = [p.label for p in parts if not p.comp]
    if unknown:
        raise Skip("the composition of " + ", ".join(unknown) + " is not known (no atom table in the output "
                   "or the database)")


def _lowest(parts: List[R.Part], per) -> Optional[R.Part]:
    order = {"OPT": 0, "SP": 1, "FREQ": 2}
    best = sorted(parts, key=lambda p: (round(p.energy / per(p), 8), order.get(p.calc_type or "", 3)))
    return best[0] if best else None


# ------------------------------------------------------------------ formation energy
def _find_elements(ctx, conn):
    if conn is None:
        return []
    run = R.Part.of_run(ctx)
    if not run.comp:
        raise Skip("the run's composition is not known")
    els = set(run.comp)
    # a bulk or a molecule of the element alone (a slab or a polymer is no standard state); not
    # another run of the run's own material (its OPT, SP and FREQ are one phase), nor of its
    # structure under another name or in a phonon supercell (R.same_structure)
    pool = R.candidates(conn, dims=("CRYSTAL", "MOLECULE"), formula_ok=lambda c: len(c) == 1 and set(c) <= els,
                        role="elements", exclude=[run.calc_id] if run.calc_id else ())
    pool = [p for p in pool if p.comp and len(p.comp) == 1 and set(p.comp) <= els and p.dim in (0, 3)
            and not (run.material_id and p.material_id == run.material_id) and not R.same_structure(run, p)]
    if len(run.comp) == 1 and run.energy is not None and run.dim in (0, 3):
        pool.append(run)                  # an element: against its lowest-energy phase
    found, missing = [], []
    for el in sorted(els):
        ok = [p for p in pool if p is run] + R.consistent(run, [p for p in pool if p is not run and set(p.comp) == {el}])
        best = _lowest(ok, lambda p: p.atoms)
        if best is None:
            missing.append(el)
        elif best is run:
            raise Skip(f"the run is the lowest-energy {el} run of the database ({R.describe_settings(run)}): its "
                       "formation energy is zero by definition")
        else:
            found.append(best.as_ref())
    if missing:
        raise Skip(f"no elemental reference run for {', '.join(missing)} in the database with "
                   f"{R.describe_settings(run)} (--reference elements=FILE|CALC_ID, or Add Reference… in the "
                   "Properties tab)")
    return found


@derived("formation_energy", calc_types=TYPES, category="energy", group=GROUP, version="1",
         references=(Reference("elements", calc_types=TYPES, many=True, find=_find_elements,
                               description="one run per element of the compound: a bulk or molecule of the "
                                           "element alone (the database's lowest energy per atom, same "
                                           "functional and basis)"),),
         outputs=_outputs("formation", "f", (("ev_per_atom", "eV/atom"),), "formation energy per atom",
                          "formation energy: the elemental references, their energies per atom, the run's "
                          "composition, warnings, and why ΔH/ΔG are missing"),
         source="the run's energy and composition, and its elemental reference runs")
def _formation(ctx):
    """Formation energy per atom against elemental reference runs."""
    run = R.Part.of_run(ctx)
    refs = _parts(ctx, "elements")
    _need_comp([run] + refs)
    by_el: Dict[str, R.Part] = {}
    for p in refs:
        if len(p.comp) != 1:
            raise Skip(f"{p.label} is not an elemental run ({R.formula(p.comp)})")
        el = next(iter(p.comp))
        if el in by_el and by_el[el].calc_id != p.calc_id:
            raise Skip(f"two references for {el}: {by_el[el].label} and {p.label}")
        by_el[el] = p
    missing = sorted(set(run.comp) - set(by_el))
    if missing:
        raise Skip(f"no elemental reference for {', '.join(missing)} (--reference elements=FILE|CALC_ID)")
    used = [by_el[el] for el in sorted(run.comp)]
    unused = [p.label for p in refs if next(iter(p.comp)) not in run.comp]
    parts = [run] + used
    warnings = R.guard(parts)
    n = run.atoms

    def fn(e):
        tot = e[0] - sum(run.comp[p.comp.most_common(1)[0][0]] * ei / p.atoms for p, ei in zip(used, e[1:]))
        return {"ev_per_atom": tot * R.HARTREE_EV / n}
    forms, thermo = R.forms(parts, fn, ctx.db)
    extra = {"composition": R.formula(run.comp), "atoms": n,
             "reference_energy_per_atom_au": {next(iter(p.comp)): p.energy / p.atoms for p in used}}
    if unused:
        extra["ignored_references"] = unused
    detail = _detail("ΔE_f = [E(run) − Σ_i n_i E(ref_i)/N(ref_i)] / N(run)", parts, warnings, extra, {}, thermo)
    return _store("formation", forms, detail)


# ------------------------------------------------------------------ cohesive energy
def _find_atoms(ctx, conn):
    if conn is None:
        return []
    run = R.Part.of_run(ctx)
    if not run.comp:
        raise Skip("the run's composition is not known")
    els = set(run.comp)
    pool = R.candidates(conn, dims=("MOLECULE",), formula_ok=lambda c: len(c) == 1 and set(c) <= els, role="atoms")
    pool = [p for p in pool if p.atoms == 1 and p.comp and set(p.comp) <= els and (p.dim in (0, None))]
    found, missing = [], []
    for el in sorted(els):
        best = _lowest(R.consistent(run, [p for p in pool if set(p.comp) == {el}]), lambda p: 1)
        if best is None:
            missing.append(el)
        else:
            found.append(best.as_ref())
    if missing:
        raise Skip(f"no isolated-atom run for {', '.join(missing)} in the database with {R.describe_settings(run)} "
                   "(--reference atoms=FILE|CALC_ID)")
    return found


@derived("cohesive_energy", calc_types=TYPES, category="energy", group=GROUP, version="1",
         references=(Reference("atoms", calc_types=TYPES, many=True, find=_find_atoms,
                               description="one isolated-atom (0D, one atom) run per element (the lowest energy: "
                                           "its ground spin state)"),),
         outputs=_outputs("cohesive", "coh", (("ev_per_atom", "eV/atom"),), "cohesive energy per atom",
                          "cohesive energy (positive: bound): the isolated-atom runs, their energies, warnings, "
                          "and why ΔH/ΔG are missing"),
         source="the run's energy and composition, and isolated-atom runs")
def _cohesive(ctx):
    """Cohesive energy per atom (positive when bound) against isolated-atom runs."""
    run = R.Part.of_run(ctx)
    if (run.dim or 0) == 0 and run.atoms <= 1:
        raise Skip("an isolated atom")
    atoms = _parts(ctx, "atoms")
    _need_comp([run] + atoms)
    by_el: Dict[str, R.Part] = {}
    for p in atoms:
        if p.atoms != 1:
            raise Skip(f"{p.label} is not an isolated atom ({R.formula(p.comp)})")
        el = next(iter(p.comp))
        if el in by_el and by_el[el].calc_id != p.calc_id:
            raise Skip(f"two isolated-atom runs for {el}: {by_el[el].label} and {p.label}")
        by_el[el] = p
    missing = sorted(set(run.comp) - set(by_el))
    if missing:
        raise Skip(f"no isolated-atom run for {', '.join(missing)} (--reference atoms=FILE|CALC_ID)")
    used = [by_el[el] for el in sorted(run.comp)]
    parts = [run] + used
    warnings = R.guard(parts)
    n = run.atoms

    def fn(e):
        tot = sum(run.comp[next(iter(p.comp))] * ei for p, ei in zip(used, e[1:])) - e[0]
        return {"ev_per_atom": tot * R.HARTREE_EV / n}
    forms, thermo = R.forms(parts, fn, ctx.db)
    detail = _detail("E_coh = [Σ_i n_i E(atom_i) − E(run)] / N(run) (positive: bound)", parts, warnings,
                     {"composition": R.formula(run.comp), "atoms": n}, {}, thermo)
    return _store("cohesive", forms, detail)


# ------------------------------------------------------------------ surface energy
def _find_bulk(ctx, conn):
    run = R.Part.of_run(ctx)
    if run.dim != 2:
        raise Skip("not a slab (2D) run")
    if conn is None:
        return []
    if not run.comp:
        raise Skip("the slab's composition is not known")
    want = R.reduced(run.comp)
    pool = R.candidates(conn, dims=("CRYSTAL",), formula_ok=lambda c: R.reduced(c) == want, role="bulk")
    pool = [p for p in pool if p.dim == 3 and p.comp and R.ratio(run.comp, p.comp)]
    best = _lowest(R.consistent(run, pool), lambda p: p.atoms)
    if best is None:
        raise Skip(f"no bulk (3D) run of {R.formula(want)} in the database with {R.describe_settings(run)} "
                   "(--reference bulk=FILE|CALC_ID)")
    return [best.as_ref()]


@derived("surface_energy", calc_types=TYPES, category="energy", group=GROUP, version="1",
         references=(Reference("bulk", calc_types=TYPES, find=_find_bulk,
                               description="a bulk (3D) run of the slab's composition (the lowest energy per atom "
                                           "of the database's, same functional and basis)"),),
         outputs=_outputs("surface", "surf", (("j_m2", "J/m²"), ("ev_ang2", "eV/Å²")), "surface energy",
                          "surface energy: the bulk run, n (bulk cells in the slab), the slab area, warnings, "
                          "and why ΔH/ΔG are missing"),
         source="the slab's energy, composition and area, and a bulk run")
def _surface(ctx):
    """Surface energy of a slab: (E_slab - n E_bulk) / 2A."""
    run = R.Part.of_run(ctx)
    if run.dim != 2:
        raise Skip("not a slab (2D) run")
    bulk = _parts(ctx, "bulk")[0]
    _need_comp([run, bulk])
    if bulk.dim != 3:
        raise Skip(f"{bulk.label} is not a bulk (3D) run")
    n = R.ratio(run.comp, bulk.comp)
    if n is None:
        raise Skip(f"the slab ({R.formula(run.comp)}) and the bulk ({R.formula(bulk.comp)}) are not commensurate: "
                   "a non-stoichiometric slab needs chemical potentials (not covered)")
    area = run.cell
    if not area:
        raise Skip("the slab's area is not known")
    warnings = R.guard([run, bulk])

    def fn(e):
        d = e[0] - n * e[1]
        return {"j_m2": d * R.HARTREE_J / (2 * area * 1e-20), "ev_ang2": d * R.HARTREE_EV / (2 * area)}
    forms, thermo = R.forms([run, bulk], fn, ctx.db)
    extra = {"n_bulk_cells": n, "area_ang2": area, "slab": R.formula(run.comp), "bulk": R.formula(bulk.comp),
             "assumes": "two equivalent surfaces (a symmetric slab)"}
    detail = _detail("γ = [E(slab) − n E(bulk)] / 2A", [run, bulk], warnings, extra, {}, thermo)
    return _store("surface", forms, detail)


# ------------------------------------------------------------------ adsorption energy
def _k(complex_: R.Part, surface: R.Part, ads: R.Part) -> Optional[int]:
    """k with complex = surface + k·adsorbate (an integer ≥ 1), else None."""
    if not (complex_.comp and surface.comp and ads.comp):
        return None
    diff = complex_.comp.copy()
    diff.subtract(surface.comp)
    if any(v < 0 for v in diff.values()):
        return None
    diff = +diff
    k = R.ratio(diff, ads.comp) if diff else None
    return int(round(k)) if k and abs(k - round(k)) < 1e-9 and round(k) >= 1 else None


def _same_cell(a: R.Part, b: R.Part) -> bool:
    return bool(a.cell and b.cell and abs(a.cell - b.cell) <= R.CELL_TOL * a.cell)


def _ads_pool(ctx, conn, run):
    surf = [p for p in R.candidates(conn, dims=("SLAB",), formula_ok=lambda c: set(c) <= set(run.comp), role="surface",
                                    exclude=[run.calc_id] if run.calc_id else ())
            if p.dim == 2 and p.comp and _same_cell(run, p)]
    ads = [p for p in R.candidates(conn, dims=("MOLECULE", "SLAB"), formula_ok=lambda c: set(c) <= set(run.comp),
                                   role="adsorbate", exclude=[run.calc_id] if run.calc_id else ())
           if p.comp and (p.dim == 0 or (p.dim == 2 and _same_cell(run, p)))]
    return R.consistent(run, surf), R.consistent(run, ads)


def _rank(run, s, a):
    shared = bool(set(s.comp) & set(a.comp))
    mis = abs((s.cell or 0) - run.cell) + (abs((a.cell or 0) - run.cell) if a.dim == 2 else 0)
    return (shared, round(mis, 6))


def _pair(ctx, conn):
    cache = ctx.inputs.setdefault("_adsorption_pair", {})
    if "pair" in cache:
        return cache["pair"]
    run = R.Part.of_run(ctx)
    if run.dim != 2:
        raise Skip("the surface and adsorbate are looked up for a slab (2D) complex only: give them "
                   "(--reference surface=FILE|CALC_ID --reference adsorbate=FILE|CALC_ID)")
    if conn is None:
        cache["pair"] = None
        return None
    if not run.comp:
        raise Skip("the complex's composition is not known")
    given_s = [R.Part.of_ref(r, "surface", conn) for r in ctx.references.get("surface") or []]
    given_a = [R.Part.of_ref(r, "adsorbate", conn) for r in ctx.references.get("adsorbate") or []]
    surf, ads = _ads_pool(ctx, conn, run)
    surf = given_s or surf
    ads = given_a or ads
    pairs = [(s, a) for s in surf for a in ads if _k(run, s, a)]
    if not pairs:
        raise Skip(f"no clean surface and adsorbate in the database that add up to {R.formula(run.comp)} in the "
                   f"same cell with {R.describe_settings(run)} (--reference surface=... --reference adsorbate=...)")
    best = sorted(pairs, key=lambda sa: _rank(run, *sa))[0]
    cache["pair"] = best
    return best


def _find_surface(ctx, conn):
    pair = _pair(ctx, conn)
    return [pair[0].as_ref()] if pair else []


def _find_adsorbate(ctx, conn):
    pair = _pair(ctx, conn)
    return [pair[1].as_ref()] if pair else []


@derived("adsorption_energy", calc_types=TYPES, category="energy", group=GROUP, version="1",
         references=(Reference("surface", calc_types=TYPES, find=_find_surface,
                               description="the clean surface (or host) in the complex's cell"),
                     Reference("adsorbate", calc_types=TYPES, find=_find_adsorbate,
                               description="the adsorbate (a molecule, or a layer in the complex's cell)")),
         outputs=_outputs("adsorption", "ads", (("ev", "eV"), ("kj_mol", "kJ/mol")), "adsorption energy per adsorbate",
                          "adsorption energy: the surface and adsorbate runs, k (adsorbates in the complex), "
                          "warnings, and why ΔH/ΔG are missing"),
         source="the complex's energy and composition, a clean-surface run and an adsorbate run")
def _adsorption(ctx):
    """Adsorption energy per adsorbate: [E(complex) - E(surface) - k E(adsorbate)] / k."""
    run = R.Part.of_run(ctx)
    surface = _parts(ctx, "surface")[0]
    ads = _parts(ctx, "adsorbate")[0]
    _need_comp([run, surface, ads])
    k = _k(run, surface, ads)
    if not k:
        raise Skip(f"{run.label} ({R.formula(run.comp)}) is not {surface.label} ({R.formula(surface.comp)}) plus a "
                   f"whole number of {ads.label} ({R.formula(ads.comp)})")
    if surface.dim != run.dim:
        raise Skip(f"{surface.label} is {surface.dim}D and the complex {run.dim}D")
    warnings = []
    if run.dim in (1, 2):
        if not _same_cell(run, surface):
            raise Skip(f"the cells differ: {run.label} {run.cell:g}, {surface.label} {surface.cell or 0:g} "
                       f"({'Å²' if run.dim == 2 else 'Å'}): the surface must be in the complex's cell")
    elif run.dim == 3 and run.cell and surface.cell and abs(run.cell - surface.cell) > R.CELL_TOL * run.cell:
        warnings.append(f"the cell volumes differ by more than {R.CELL_TOL:.0%}: {run.label} {run.cell:g} Å³, "
                        f"{surface.label} {surface.cell:g} Å³")
    if ads.dim not in (0, None) and not (ads.dim == run.dim and _same_cell(run, ads)):
        raise Skip(f"{ads.label} is neither a molecule nor a layer in the complex's cell")
    warnings = R.guard([run, surface, ads]) + warnings

    def fn(e):
        d = (e[0] - e[1] - k * e[2]) / k
        return {"ev": d * R.HARTREE_EV, "kj_mol": d * R.HARTREE_EV * R.EV_KJ_MOL}
    forms, thermo = R.forms([run, surface, ads], fn, ctx.db)
    extra = {"k_adsorbates": k, "complex": R.formula(run.comp), "surface": R.formula(surface.comp),
             "adsorbate": R.formula(ads.comp)}
    detail = _detail("E_ads = [E(complex) − E(surface) − k E(adsorbate)] / k (negative: binds)",
                     [run, surface, ads], warnings, extra, {}, thermo)
    return _store("adsorption", forms, detail)


# ------------------------------------------------------------------ reaction energy
@derived("reaction_energy", calc_types=TYPES, category="energy", group=GROUP, version="1",
         references=(Reference("reactants", calc_types=TYPES, many=True,
                               description="the reactant runs (N*FILE|N*CALC_ID for a coefficient)"),
                     Reference("products", calc_types=TYPES, many=True, required=False,
                               description="the other product runs; the run itself is a product")),
         outputs=_outputs("reaction", "r", (("ev", "eV"), ("kj_mol", "kJ/mol")), "reaction energy",
                          "reaction energy: the equation, each run's coefficient and energy, warnings, and why "
                          "ΔH/ΔG are missing"),
         source="the run (a product) and the reactant and product runs the user gives")
def _reaction(ctx):
    """Reaction energy Σ products - Σ reactants (the run is a product)."""
    run = R.Part.of_run(ctx)
    prods = _parts(ctx, "products")
    reacts = _parts(ctx, "reactants")
    weight = 1.0
    others = []
    for p in prods:
        same = (p.calc_id and p.calc_id == run.calc_id) or (
            p.output_file and run.output_file and p.output_file.resolve() == run.output_file.resolve())
        if same:
            weight = p.weight if p.weight is not None else 1.0
        else:
            others.append(p)
    for p in reacts:
        if (p.calc_id and p.calc_id == run.calc_id) or (p.material_id and p.material_id == run.material_id):
            raise Skip(f"{p.label} is a run of the run's own material: not a reactant")
    parts = [run] + others + reacts
    signs = [1] + [1] * len(others) + [-1] * len(reacts)
    given = [weight if any(p.weight is not None for p in prods + reacts) else None] + [p.weight for p in others + reacts]
    warnings = R.guard(parts)
    comps = [p.comp for p in parts]
    signed = list(zip(signs, comps))
    if all(w is None for w in given):
        if not all(comps):
            raise Skip("the coefficients cannot be balanced without every run's composition: give them "
                       "(--reference reactants=2*FILE)")
        coef = R.balance(signed)
        if coef is None:
            left = R.imbalance(signed, [1.0] * len(parts))
            raise Skip("the reaction cannot be balanced in exactly one way with positive coefficients"
                       + (f" (at 1:1, left over: {left})" if left else "") + ": give the coefficients "
                       "(--reference reactants=2*FILE)")
        coef = [float(c) for c in coef]
    else:
        coef = [w if w is not None else 1.0 for w in given]
        if all(comps):
            left = R.imbalance(signed, coef)
            if left:
                raise Skip("the reaction is not balanced: " + ", ".join(f"{e} {v:+g}" for e, v in sorted(left.items()))
                           + " (products minus reactants)")
        else:
            warnings.append("the balance is not checked: a run's composition is not known")

    def fn(e):
        d = sum(s * c * ei for s, c, ei in zip(signs, coef, e))
        return {"ev": d * R.HARTREE_EV, "kj_mol": d * R.HARTREE_EV * R.EV_KJ_MOL}
    forms, thermo = R.forms(parts, fn, ctx.db)
    side = lambda sgn: " + ".join(f"{R.fmt_coef(c)} {R.formula(p.comp)}" for p, s, c in zip(parts, signs, coef)
                                  if s == sgn)
    for p, c in zip(parts, coef):
        p.weight = c
    extra = {"equation": f"{side(-1)} → {side(1)}", "per": "the reaction as written: each coefficient counts the "
             "run's cell (a molecule: one molecule)"}
    detail = _detail("ΔE_r = Σ_products c E − Σ_reactants c E", parts, warnings, extra, {}, thermo)
    return _store("reaction", forms, detail)


# ------------------------------------------------------------------ the saved choices
@derived("reference_runs", calc_types=TYPES, category="computational", group=GROUP, interactive=True, version="1",
         outputs=(Output("reference_runs", "JSON", "json", label="Reference runs chosen", group=GROUP,
                         description="the reference runs chosen for this run's reference energies (--reference, "
                                     "the Properties tab's Add Reference…): {property: {reference: [calc_id or "
                                     "file, coefficient]}}; every extraction uses them"),),
         source="--reference ROLE=FILE|CALC_ID, or Add Reference… in the Properties tab")
def _reference_runs(ctx):
    """The reference runs chosen for this run (kept across re-extraction)."""
    choices = ctx.inputs.get("choices")
    if not isinstance(choices, dict):
        raise Skip("no choices")
    return {"reference_runs": choices}
