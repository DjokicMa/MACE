/* The Recover flow shared by the Jobs panel (a failed job, or Recover
 * Failed… for the folder) and the Workflows view (one failed step): show
 * `mace recover --dry-run`, confirm with the command a run uses, ask
 * separately about jobs that ended too long ago (the queue manager does not
 * resubmit those by itself), then run `mace recover --yes` and say what it
 * resubmitted. The server answers both routes the same way (recover.py).
 *
 * o: {preview(): the preview task's answer, run(body): the run task's answer,
 *     title: the confirm's title, one: true for one step (singular wording)} */
export async function recoverFlow(h, o) {
  const { toast } = h;
  const p = await o.preview();
  if (!p || p.error) { toast((p && p.error) || "The preview failed", "error"); return null; }
  if (!p.ready && !p.stale) {
    await h.confirm({ title: "Nothing to Resubmit", body: `mace recover found nothing it can resubmit${o.one ? " for this step" : ""}:`,
      pre: p.text, yes: "OK", no: "Close" });
    return null;
  }
  if (p.network) toast("Recovering jobs is turned off while MACE GUI is open to the network", "warning");
  const ok = await h.confirm({ title: o.title, body: `This runs: ${p.run_argv.join(" ")}`, pre: p.text,
    yes: p.ready ? (o.one ? "Resubmit" : `Resubmit ${p.ready}`) : "Continue", no: "Keep As Is", defaultNo: true });
  if (!ok) return null;
  let stale = false;
  if (p.stale) {
    stale = await h.confirm(o.one
      ? { title: "Resubmit an Old Job?", body: `It ended more than ${p.stale_days} days ago; the queue manager does not resubmit those by itself. Resubmit it anyway?`,
        yes: "Resubmit It", no: "Leave It", defaultNo: true }
      : { title: "Resubmit Old Jobs Too?", body: `${p.stale} of them ended more than ${p.stale_days} days ago; the queue manager does not resubmit those by itself. Resubmit them too?`,
        yes: "Resubmit Them", no: "Leave Them", defaultNo: true });
  }
  if (!p.ready && !stale) return null;
  const r = await o.run({ preview_id: p.preview_id, include_stale: stale });
  if (!r || r.error) { toast((r && r.error) || "mace recover failed", "error"); return r; }
  const done = (r.results || []).filter((x) => x.status === "submitted");
  toast(done.length ? `Resubmitted ${o.one ? "as job " : ""}${done.map((x) => x.job_id).join(", ")}` : "Nothing was resubmitted",
    done.length ? "success" : "warning");
  for (const x of (r.results || []).filter((y) => y.status !== "submitted")) toast(`${x.calc_id}: ${x.why}`, "warning");
  return r;
}
