# Archive Policy

This repository intentionally contains only the artifacts needed to reproduce and audit the paper-facing experiments.

The cleanup performed for the public reproducibility release moved non-paper artifacts to a local archive outside the Git tree:

`/home/izhan/CISOSE_RL_Experimentations_archive_20260509`

Archived material includes smoke runs, calibration sweeps, abandoned candidate result folders, local MLflow backend/artifacts, vendored dependencies, virtual environments, Python bytecode caches, and earlier draft-section notes. These files were not deleted; they were excluded from the public repository because they are not part of the final evidence chain and would make the release difficult to audit.

The public repository keeps:

- final experiment source code and configuration files
- final paper-facing result CSVs and run manifests
- final paper figures/tables and internal claim-audit reports
- isolated MLflow launch scripts, without committing local MLflow state
- the finalized paper PDF used to align the artifact map
