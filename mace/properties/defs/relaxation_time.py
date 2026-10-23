"""The deformation-potential relaxation time of a strain series (``mace tau
fit``, the GUI's Relaxation Time dialog): an interactive property. The user
picks the series, the carrier and what overrides the fits (E1, C, m*, T);
**Save to Database** in the dialog stores the result with the unstrained run
(rows by ``mace-gui``, which a re-extraction keeps). One set of keys per
carrier and strain axis, so electrons and holes along x, y and z stand side
by side. The strained runs are the property's references.
"""
from __future__ import annotations

from mace.properties.registry import Output, Reference, Skip, derived

CARRIERS = ("electron", "hole")
AXES = ("x", "y", "z")


def _outputs():
    out = []
    for c in CARRIERS:
        who = "electrons" if c == "electron" else "holes"
        for a in AXES:
            out += [Output(f"relaxation_time_dpt_{c}_{a}_fs", "fs", label=f"τ<sub>DPT</sub> ({who}, {a})", precision=4,
                           description=f"deformation-potential relaxation time of {who}, strain along {a}"),
                    Output(f"mobility_dpt_{c}_{a}_cm2_vs", "cm²/(V·s)", label=f"μ<sub>DPT</sub> ({who}, {a})",
                           precision=4, description=f"mobility e·τ/m_c of {who} along {a}"),
                    Output(f"relaxation_time_dpt_{c}_{a}", "JSON", "json", blob=True,
                           label=f"τ<sub>DPT</sub> details ({who}, {a})",
                           description="the full mace tau fit result: formula, inputs and their sources (E1, C, "
                                       "m*, m_d, T), the strain fit and the masses used")]
    return tuple(out)


@derived("relaxation_time_dpt", calc_types=("SP",), category="transport", group="Relaxation time",
         interactive=True, version="1", outputs=_outputs(),
         references=(Reference("strained_runs", calc_types=("SP",), many=True,
                               description="the strained SP runs of the series (mace tau strain)"),),
         source="mace tau fit on a strain series (GUI: Relaxation Time > Fit > Save to Database)")
def _relaxation_time(ctx):
    """Deformation-potential relaxation time and mobility of a strain series."""
    res = ctx.inputs.get("result") or {}
    carrier, axis = res.get("carrier"), res.get("axis")
    if carrier not in CARRIERS or axis not in AXES or res.get("tau_fs") is None:
        raise Skip("not a mace tau fit result (carrier, axis and tau_fs)")
    detail = {k: v for k, v in res.items() if k != "relaxtim"}
    return {f"relaxation_time_dpt_{carrier}_{axis}_fs": res["tau_fs"],
            f"mobility_dpt_{carrier}_{axis}_cm2_vs": res.get("mobility_cm2_Vs"),
            f"relaxation_time_dpt_{carrier}_{axis}": detail}
