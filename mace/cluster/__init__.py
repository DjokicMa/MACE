"""Cluster monitoring: `mace cluster` and the GUI's Cluster view.

Read-only, cached and rate-limited views of a SLURM cluster (nodes,
partitions, limits, quotas, reservations) and of the user's own jobs
(detail, progress, ETA, walltime risk, history estimates).

Every module here imports only the Python standard library at module level,
so `mace cluster` runs on a cluster's module Python without numpy, pandas,
PyYAML or rich. Nothing here submits, cancels, modifies or deletes anything:
the only sbatch call is ``sbatch --test-only`` (tests/test_cluster_readonly.py).
"""
