#!/usr/bin/env python3
"""Build the bundled basis library files (Crystal_d12/basis_sets/*.lib) and
PROVENANCE.md.

    python build_library.py --from-dirs DIR   # first build, from the old
                                              # per-element directories in DIR
    python build_library.py                   # rewrite the #% provenance lines
                                              # of the existing .lib files;
                                              # record bodies are copied as-is

Record bodies are never edited here: tests/data/basis_library_pins.json pins
the bytes of every record, and tests/test_basis_library_pins.py fails if a
body changes. Provenance comes from the CRYSTAL23 basis set library pages
(crystal.unito.it, matched by exact numeric comparison - same lines, same
numbers at printed precision) and from FACTS.md's audit against
basissetexchange.org. A DOI is written only when it was copied from a page.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import basis_library  # noqa: E402

SITE = "https://www.crystal.unito.it/Basis_Sets/"
# The families built from MACE's old per-element directories.
LEGACY_FAMILIES = ("full.basis.triplezeta", "full.basis.doublezeta", "stuttgart")
RETRIEVED = "2026-10-01"
UNVERIFIED = "unverified (legacy MACE file)"

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es").split()

PAGES = {
    1: "hydrogen", 3: "lithium", 4: "beryllium", 5: "boron", 6: "carbon", 7: "nitrogen",
    8: "oxygen", 9: "fluorine", 11: "sodium", 12: "magnesium", 13: "aluminium",
    14: "silicon", 15: "phosphorus", 16: "sulphur", 17: "chlorine", 19: "potassium",
    20: "calcium", 21: "scandium", 22: "titanium", 23: "vanadium", 24: "chromium",
    25: "manganese", 26: "iron", 27: "cobalt", 28: "nickel", 29: "copper", 30: "zinc",
    31: "gallium", 32: "germanium", 33: "arsenic", 34: "selenium", 35: "bromine",
    37: "rubidium", 38: "strontium", 39: "yttrium", 40: "zirconium", 41: "niobium",
    42: "molibdenum", 43: "technecium", 44: "ruthenium", 45: "rhodium", 46: "palladium",
    47: "silver", 48: "cadmium", 49: "indium", 50: "tin", 51: "antimony", 52: "tellurium",
    53: "iodine", 55: "caesium", 56: "barium", 57: "lanthanum", 58: "cerium",
    59: "praseodymium", 60: "neodymium", 61: "promethium", 62: "samarium", 63: "europium",
    64: "gadolinium", 65: "terbium", 66: "dysprosium", 67: "holmium", 68: "erbium",
    69: "thulium", 70: "ytterbium", 71: "lutetium", 72: "hafnium", 73: "tantalum",
    74: "tungsten", 75: "rhenium", 76: "osmium", 77: "iridium", 78: "platinum", 79: "gold",
    80: "mercury", 81: "thallium", 82: "lead", 83: "bismuth", 89: "actinium", 90: "thorium",
    91: "protactinium", 92: "uranium", 93: "neptunium", 94: "plutonium", 95: "americium",
    96: "curium", 97: "berkelium", 98: "californium", 99: "einsteinium",
}

# References and DOIs as printed on the crystal.unito.it pages.
REF_REV2_2019 = ("D. Vilela Oliveira, M. F. Peintinger, J. Laun, T. Bredow, J. Comput. Chem. "
                 "40 (2019) 2364-2376")
DOI_REV2_2019 = "10.1002/jcc.26013"          # hydrogen.html, X_pob_TZVP_rev2
REF_2018 = "J. Laun, D. V. Oliveira, T. Bredow, J. Comput. Chem. 39 (2018) 1285-1290"
DOI_2018 = "10.1002/jcc.25195"               # rubidium.html, Rb_POB_TZVP_2018
REF_REV2_6TH = ("J. Laun, T. Bredow, J. Comput. Chem. (2021), \"BSSE-corrected consistent "
                "Gaussian basis sets of triple-zeta valence with polarization quality of the "
                "sixth period for solid-state calculations\"")
DOI_REV2_6TH = "10.1002/jcc.26521"           # caesium.html / hafnium.html
REF_ERD = ("Kh. E. El-Kelany, C. Ravoux, J. K. Desmarais, Y. Pan, J. S. Tse, R. Dovesi, "
           "A. Erba (in preparation); J. K. Desmarais, A. Erba, R. Dovesi, Theor. Chem. Acc. "
           "(2018) (as printed on the site, no DOI given)")
REF_TC = "\"Optimized basis set for Tc\" (as printed on the site, no reference given)"


def z_of(key):
    n = int(key.split(".")[0])
    return n - 200 if n > 200 else n


def url(z, anchor):
    return f"{SITE}{PAGES[z]}.html#{anchor}"


def site_match(z, anchor, reference, doi, extra=None):
    meta = [("source", anchor), ("reference", reference)]
    if doi:
        meta.append(("doi", doi))
    meta.append(("url", url(z, anchor)))
    note = (f"exact numeric match to {anchor} on crystal.unito.it "
            f"(retrieved {RETRIEVED})")
    if extra:
        note += "; " + extra
    meta.append(("provenance", note))
    return meta


def triplezeta_provenance(key):
    z = z_of(key)
    sym = SYMBOLS[z - 1]
    if z == 7:
        return [("source", "pob-TZVP-rev2"), ("reference", REF_REV2_2019),
                ("doi", DOI_REV2_2019), ("url", url(7, "N_pob_TZVP_2012")),
                ("provenance", "FACTS.md audit (basissetexchange.org): pob-TZVP-rev2; "
                               "numerically identical to N_pob_TZVP_2012 on crystal.unito.it, "
                               "which lists no N_pob_TZVP_rev2")]
    if z == 14:
        return [("source", "pob-TZVP-rev2"), ("reference", REF_REV2_2019),
                ("doi", DOI_REV2_2019),
                ("provenance", "FACTS.md audit (basissetexchange.org): pob-TZVP-rev2; "
                               "crystal.unito.it lists only Si_pob_TZVP_2012, which differs")]
    if z <= 35:
        return site_match(z, f"{sym}_pob_TZVP_rev2", REF_REV2_2019, DOI_REV2_2019)
    if z == 43:
        return site_match(43, "Tc_ERD_2017", REF_TC, None, "all-electron record")
    if 37 <= z <= 53:
        return site_match(z, f"{sym}_POB_TZVP_2018", REF_2018, DOI_2018,
                          "crystal.unito.it also has the newer "
                          f"{sym}_POB_TZVP_rev2 (Laun & Bredow 2022)")
    if z == 56:
        return [("source", "pob-TZVP-rev2"), ("reference", REF_REV2_6TH),
                ("doi", DOI_REV2_6TH),
                ("provenance", "FACTS.md audit (basissetexchange.org): pob-TZVP-rev2; "
                               "crystal.unito.it lists only Ba_pob_TZVP_rev2_SOC")]
    if z == 55 or 72 <= z <= 83:
        return site_match(z, f"{sym}_pob_TZVP_rev2", REF_REV2_6TH, DOI_REV2_6TH)
    if 57 <= z <= 71:
        return site_match(z, f"{sym}_ERD_SMALL_CORE_PP_4F_IN_VALENCE_NO_G_2017", REF_ERD, None,
                          "small-core pseudopotential, 4f in valence")
    if 89 <= z <= 99:
        anchor = f"{sym}_ERD_SMALL_CORE_PP_4F_IN_VALENCE_NO_G_2017"
        note = ("FACTS.md audit: ERD 2017. Exponents and coefficients equal "
                f"{anchor} on crystal.unito.it; the f shell's formal charge (CHE) is "
                "3 lower than the site's")
        if z == 94:
            note += ("; the first line said 11 shells for the 9 the record has "
                     "(corrected to 9, as the site prints it)")
        return [("source", anchor), ("reference", REF_ERD), ("url", url(z, anchor)),
                ("provenance", note)]
    raise KeyError(f"no provenance rule for full.basis.triplezeta record {key}")


def inherited_or_unverified(body, tz_records, family):
    """Provenance of a DZ / stuttgart record: a byte-identical copy of a TZ
    record inherits the TZ label, everything else is unverified."""
    for tz_key, tz_body in tz_records.items():
        if body == tz_body:
            meta = [m for m in triplezeta_provenance(tz_key) if m[0] != "provenance"]
            prov = dict(triplezeta_provenance(tz_key))["provenance"]
            meta.append(("provenance", f"byte-identical to full.basis.triplezeta record "
                                       f"{tz_key} ({prov})"))
            return meta
    return [("provenance", UNVERIFIED)]


DESCRIPTIONS = {
    "full.basis.triplezeta": [
        "The external basis records MACE has always written for TZVP-REV2 (menu option 2).",
        "Mixed provenance, per record below: pob-TZVP-rev2 for H-Br, Cs, Ba and Hf-Bi;",
        "pob-TZVP 2018 for Rb-I except Tc (ERD 2017, all-electron); ERD small-core 2017",
        "(4f in valence) for La-Lu; ERD 2017 for Ac-Es. No noble gases, Po, At, Fr, Ra.",
    ],
    "full.basis.doublezeta": [
        "The external basis records MACE has always written for DZVP-REV2 (menu option 1).",
        "41 records (Tc, Cs-Bi, Ac-Es) are byte-identical to the triplezeta family's.",
    ],
    "stuttgart": [
        "Legacy Stuttgart-type records shipped with MACE. Not written into decks by the",
        "standard basis menus; read by the SOC tooling. 250 and 251 are byte-identical",
        "to the triplezeta family's records.",
    ],
}


def _sort_key(key):
    head, _, tail = key.partition(".")
    return (int(head), tail)


def render(family, description, records, provenance, missing=None):
    """Bytes of a library file."""
    out = [f"{basis_library.FORMAT_HEADER} {basis_library.FORMAT_VERSION}\n",
           f"#%family {family}\n"]
    out += [f"#%description {line}\n" for line in description]
    for z, reason in sorted((missing or {}).items(), key=lambda kv: int(kv[0])):
        out.append(f"#%missing {z} {reason}\n")
    data = "".join(out).encode("ascii")
    for key in sorted(records, key=_sort_key):
        body = records[key]
        head = f"#%record {key} bytes={len(body)}\n"
        head += "".join(f"#%  {f} {v}\n" for f, v in provenance[key])
        data += head.encode("ascii") + body + b"\n"
    return data


def read_dirs(root):
    out = {}
    for family in LEGACY_FAMILIES:
        d = os.path.join(root, family)
        out[family] = {name: open(os.path.join(d, name), "rb").read()
                       for name in os.listdir(d)
                       if name != "__init__.py" and os.path.isfile(os.path.join(d, name))}
    return out


def read_libs():
    return {family: dict(basis_library.iter_records(family))
            for family in LEGACY_FAMILIES}


def provenance_for(family, records, tz_records):
    if family == "full.basis.triplezeta":
        return {k: triplezeta_provenance(k) for k in records}
    if family == "full.basis.doublezeta":
        return {k: doublezeta_provenance(k, records[k], tz_records) for k in records}
    return {k: inherited_or_unverified(records[k], tz_records, family) for k in records}


def doublezeta_provenance(key, body, tz_records):
    """Copies of TZ records inherit its label. The others were compared with
    the site's X_pob_DZVP_rev2 (H-Br) and X_POB_DZVP_2018 (Rb-I) records:
    exact numeric equality, same lines and numbers at printed precision
    (build_library.py --verify). Si has no match there and stays unverified."""
    meta = inherited_or_unverified(body, tz_records, "full.basis.doublezeta")
    if meta != [("provenance", UNVERIFIED)]:
        return meta
    z = z_of(key)
    sym = SYMBOLS[z - 1]
    if z <= 35 and z != 14:
        return site_match(z, f"{sym}_pob_DZVP_rev2", REF_REV2_2019, DOI_REV2_2019)
    if 37 <= z <= 53 and z != 43:
        return site_match(z, f"{sym}_POB_DZVP_2018", REF_2018, DOI_2018)
    return meta


def provenance_markdown(tables):
    lines = ["# Bundled basis set provenance", "",
             "Generated by build_library.py from the `#%` lines of the .lib files; "
             "do not edit by hand.", ""]
    for family, (records, provenance, missing) in tables.items():
        lines += [f"## {family}", "", "| record | Z | element | source | DOI | provenance |",
                  "|---|---|---|---|---|---|"]
        for key in sorted(records, key=_sort_key):
            meta = dict()
            for f, v in provenance[key]:
                meta.setdefault(f, []).append(v)
            z = z_of(key)
            lines.append("| {} | {} | {} | {} | {} | {} |".format(
                key, z, SYMBOLS[z - 1] if z <= len(SYMBOLS) else "?",
                "; ".join(meta.get("source", ["-"])),
                "; ".join(meta.get("doi", ["-"])),
                "; ".join(meta.get("provenance", ["-"])).replace("|", "/")))
        for z, reason in sorted((missing or {}).items(), key=lambda kv: int(kv[0])):
            lines.append(f"| - | {z} | {SYMBOLS[int(z) - 1]} | not stored | - | {reason} |")
        lines.append("")
    return "\n".join(lines)


def build_bundled(root=None):
    """{family: library bytes} and the tables PROVENANCE.md is made from."""
    sources = read_dirs(root) if root else read_libs()
    tz = sources["full.basis.triplezeta"]
    files, tables = {}, {}
    for family in LEGACY_FAMILIES:
        records = sources[family]
        provenance = provenance_for(family, records, tz)
        files[family] = render(family, DESCRIPTIONS[family], records, provenance)
        tables[family] = (records, provenance, {})
    return files, tables


PROVENANCE_MD = os.path.join(HERE, "PROVENANCE.md")


# --- fetched families (need the crystal.unito.it pages: --fetch CACHE) -------

REV2022 = "full.basis.triplezeta.rev2022"
REV2022_ELEMENTS = list(range(37, 43)) + list(range(44, 54)) + list(range(57, 72))
REV2022_DESCRIPTION = [
    "Opt-in update of full.basis.triplezeta (menu option 4, external_basis_family",
    "\"tz-rev2022\"). Rb-Mo and Ru-I use X_POB_TZVP_rev2 (Laun & Bredow, J. Comput.",
    "Chem. 2022, 43, 839) and La-Lu use X_pob_TZVP_rev2 (Seidler, Laun & Bredow,",
    "J. Comput. Chem. 2023) in place of pob-TZVP 2018 and ERD small-core 2017. Tc keeps",
    "the ERD 2017 all-electron record (the site has no Tc_POB_TZVP_rev2); every other",
    "record is byte-identical to full.basis.triplezeta.",
    "Core treatment: Rb-I keep the same INPUT ECP as pob-TZVP 2018 (same ZNUC). Ce-Lu",
    "use a 28-electron core with 4f in the valence (ZNUC = Z - 28), the same core size",
    "as the ERD small-core records. La differs: ZNUC = 11, a 46-electron core, where",
    "ERD has ZNUC = 29 - each La atom carries 18 fewer electrons than with",
    "full.basis.triplezeta, which shows in the Mulliken NO. OF ELECTRONS.",
]

SOC = "pob-tzvp-rev2-soc"
SOC_ELEMENTS = list(range(37, 43)) + list(range(44, 54)) + [55, 56] + list(range(72, 84))
SOC_DESCRIPTION = [
    "X_pob_TZVP_rev2_SOC records from crystal.unito.it, for two-component SCF with",
    "spin-orbit coupling (CRYSTAL23 manual pp.169-175): the pob-TZVP-rev2 valence shells",
    "(numerically identical to the X_POB_TZVP_rev2 / X_pob_TZVP_rev2 records, checked)",
    "on CRYSTAL's internal STUTSC library, the Stuttgart-Cologne small-core fully",
    "relativistic ECP (manual p.84), in place of the scalar INPUT ECP. Rb-Mo, Ru-I, Cs,",
    "Ba and Hf-Bi. Data only: never offered as an ordinary external basis; read with",
    "basis_library.soc_basis_record(z). Not bundled: the site's special X_*_SOC sets",
    "(e.g. Bi_4s3p3d_SOC, Cs_5s5p2d_SOC) and Po_pob_TZVP_rev2_SOC (no Po in",
    "full.basis.triplezeta).",
]


def fetched_meta(found, extra_note):
    import fetch_unito

    meta = [("source", found["anchor"]),
            ("reference", fetch_unito.ascii_text(found["reference"]))]
    meta += [("doi", d) for d in found["dois"]]
    meta += [("url", found["url"]), ("site", fetch_unito.SITE_LABEL),
             ("retrieved", found["retrieved"]), ("page_sha256", found["page_sha256"]),
             ("core", found["core"]),
             ("provenance", "fetched from crystal.unito.it and cut by the CRYSTAL23 "
                            f"basis grammar; label {found['label']} = 200 + Z checked"
                            + (f"; {extra_note}" if extra_note else ""))]
    return meta


def fetch_sets(elements, set_name_for, cache):
    """{z: found-dict} for every element that passed, {z: reason} otherwise."""
    import fetch_unito

    good, missing = {}, {}
    for z in elements:
        try:
            page = fetch_unito.fetch_page(z, cache)
            found = fetch_unito.get_set(page, set_name_for(z))
        except (LookupError, fetch_unito.GrammarError) as exc:
            missing[str(z)] = fetch_unito.ascii_text(str(exc))
            continue
        problem = fetch_unito.check_ecp_set(z, found)
        if problem:
            missing[str(z)] = fetch_unito.ascii_text(f"{found['anchor']}: {problem}")
            continue
        good[z] = found
    return good, missing


def _znuc(record):
    lines = record.split("\n")
    return lines[2].split()[0] if lines[1].strip().upper() == "INPUT" else None


def build_rev2022(cache):
    import fetch_unito

    tz = dict(basis_library.iter_records("full.basis.triplezeta"))
    tz_prov = {k: basis_library.load_family("full.basis.triplezeta").provenance(k) for k in tz}
    good, missing = fetch_sets(
        REV2022_ELEMENTS, lambda z: f"{fetch_unito.symbol(z)}_pob_TZVP_rev2", cache)
    records, provenance = dict(tz), dict(tz_prov)
    for z in REV2022_ELEMENTS:
        key = str(200 + z)
        if str(z) in missing:
            # Never fall back silently to the older record: the element is
            # refused, with the reason, until the data can be fetched.
            records.pop(key, None)
            provenance.pop(key, None)
            continue
        found = good[z]
        old_znuc, new_znuc = _znuc(tz[key].decode("ascii")), _znuc(found["record"])
        note = (f"replaces full.basis.triplezeta {key} "
                f"({dict(tz_prov[key])['source']}, ZNUC {old_znuc}); ZNUC {new_znuc}")
        if old_znuc != new_znuc:
            note += " - DIFFERENT CORE from full.basis.triplezeta"
        records[key] = found["record"].encode("ascii")
        provenance[key] = fetched_meta(found, note)
    provenance["43"] = list(provenance["43"]) + [
        ("note", "kept: crystal.unito.it lists no Tc_POB_TZVP_rev2 (checked "
                 f"{RETRIEVED}); CRYSTAL23's internal POB-TZVP-REV2 has no Tc either")]
    return render(REV2022, REV2022_DESCRIPTION, records, provenance, missing), \
        (records, provenance, missing)


def build_soc(cache):
    import fetch_unito

    good, missing = fetch_sets(
        SOC_ELEMENTS, lambda z: f"{fetch_unito.symbol(z)}_pob_TZVP_rev2_SOC", cache)
    records, provenance = {}, {}
    for z, found in good.items():
        if found["record"].split("\n")[1].strip().upper() != "STUTSC":
            missing[str(z)] = f"{found['anchor']}: ECP is not STUTSC"
            continue
        records[str(200 + z)] = found["record"].encode("ascii")
        provenance[str(200 + z)] = fetched_meta(found, "STUTSC kept exactly as printed")
    return render(SOC, SOC_DESCRIPTION, records, provenance, missing), \
        (records, provenance, missing)


FETCHED_FAMILIES = (REV2022, SOC)


def verify_against_triplezeta(cache):
    """The one-off check behind the rev2022 family: the site's rev2 records
    for H-Br, Cs, Ba and Hf-Bi must equal the bundled TZ records."""
    import fetch_unito

    tz = basis_library.load_family("full.basis.triplezeta")
    lines = []
    for z in [z for z in range(1, 36)] + [55, 56] + list(range(72, 84)):
        key = str(200 + z) if z > 36 else str(z)
        if key not in tz:
            continue
        try:
            page = fetch_unito.fetch_page(z, cache)
            found = fetch_unito.get_set(page, f"{fetch_unito.symbol(z)}_pob_TZVP_rev2")
        except (LookupError, fetch_unito.GrammarError) as exc:
            lines.append(f"{fetch_unito.symbol(z):>2} {key}: not checkable ({exc})")
            continue
        same = fetch_unito.same_numbers(found["record"], tz.record(key))
        lines.append(f"{fetch_unito.symbol(z):>2} {key}: "
                     f"{'equal' if same else 'DIFFERENT'} ({found['anchor']})")
    dz = basis_library.load_family("full.basis.doublezeta")
    lines.append("full.basis.doublezeta against X_pob_DZVP_rev2 / X_POB_DZVP_2018:")
    for key in dz.keys():
        z = z_of(key)
        if z > 53 or z == 43:
            continue
        name = (f"{fetch_unito.symbol(z)}_pob_DZVP_rev2" if z <= 35
                else f"{fetch_unito.symbol(z)}_POB_DZVP_2018")
        try:
            found = fetch_unito.get_set(fetch_unito.fetch_page(z, cache), name)
        except (LookupError, fetch_unito.GrammarError) as exc:
            lines.append(f"{fetch_unito.symbol(z):>2} {key}: not checkable ({exc})")
            continue
        same = fetch_unito.same_numbers(found["record"], dz.record(key))
        lines.append(f"{fetch_unito.symbol(z):>2} {key}: "
                     f"{'equal' if same else 'DIFFERENT'} ({found['anchor']})")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--from-dirs", metavar="DIR",
                        help="build from the old per-element directories under DIR")
    parser.add_argument("--check", action="store_true",
                        help="compare with the committed files, write nothing")
    parser.add_argument("--fetch", metavar="CACHE",
                        help="(re)build the families fetched from crystal.unito.it, "
                             "caching the pages in CACHE")
    parser.add_argument("--verify", metavar="CACHE",
                        help="compare the site's rev2 records with full.basis.triplezeta")
    args = parser.parse_args(argv)
    if args.verify:
        print("\n".join(verify_against_triplezeta(args.verify)))
        return 0
    if args.fetch:
        for name, build in ((REV2022, build_rev2022), (SOC, build_soc)):
            data, (records, _prov, missing) = build(args.fetch)
            with open(basis_library.library_file(name), "wb") as fh:
                fh.write(data)
            print(f"wrote {basis_library.library_file(name)}: {len(records)} records"
                  + (f", missing {sorted(missing, key=int)}" if missing else ""))
            for z, reason in sorted(missing.items(), key=lambda kv: int(kv[0])):
                print(f"  missing {z}: {reason}")
        basis_library.load_family.cache_clear()
    files, tables = build_bundled(args.from_dirs)
    for name in FETCHED_FAMILIES:
        if os.path.isfile(basis_library.library_file(name)):
            lib = basis_library.load_file(basis_library.library_file(name))
            tables[name] = ({k: lib.record_bytes(k) for k in lib.keys()},
                            {k: lib.provenance(k) for k in lib.keys()}, lib.missing)
    markdown = provenance_markdown(tables)
    if args.check:
        stale = [f for f, data in files.items()
                 if open(basis_library.library_file(f), "rb").read() != data]
        if open(PROVENANCE_MD).read() != markdown:
            stale.append("PROVENANCE.md")
        print("up to date" if not stale else "stale: " + ", ".join(stale))
        return 1 if stale else 0
    for family, data in files.items():
        with open(basis_library.library_file(family), "wb") as fh:
            fh.write(data)
        print(f"wrote {basis_library.library_file(family)}")
    with open(PROVENANCE_MD, "w") as fh:
        fh.write(markdown)
    basis_library.load_family.cache_clear()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
