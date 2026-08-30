# Local development and GPU experiments

Codex always runs on the local Windows PC. Use this local repository for code inspection and editing, Git operations, unit tests, pytest, linting, type checking, debugging, CPU-only sanity checks, and analysis of downloaded results. Do not use the university server for ordinary development. If validation specifically requires CUDA/GPU resources, treat it as a remote experiment and use the built-in command approval mechanism before any remote operation.

## University SLURM environment

- SSH host: `slurm-login1.lnx.biu.ac.il`
- Remote repository: `/home/dsi/zusmang/TabICL/tabicl`
- Remote results: `/home/dsi/zusmang/TabICL/tabicl/results`
- Local results: `C:\Users\Gilad\Documents\Cursor\TabICL\results`

Mirror each downloaded file's path relative to remote `results/` beneath the local results directory. For example, remote `results/foo/bar/summary.json` maps to local `results\foo\bar\summary.json`. Analyze results locally after downloading them.

## Remote-access rules

- Every `ssh`, `scp`, `sftp`, or `rsync` operation requires explicit user approval for the exact command immediately before execution. Use only Codex's built-in command approval mechanism: construct the exact command, invoke it through the command execution tool, and let the normal approval UI request permission. If approved, execute immediately and continue; if denied, do not execute it.
- Never ask for remote-operation permission conversationally (for example, "May I run this command?", "Should I proceed?", or "Do you approve?"). Never request or rely on persistent, blanket, session-wide, or future approval.
- Do not bypass approval through a script, subprocess, alternate executable, or shell wrapper.
- Use normal configured access to the host. Never inspect, create, modify, move, delete, or explicitly select SSH keys or SSH configuration; do not use `-i`.
- The user controls the university VPN manually and will not provide its password. Never connect or configure the VPN, request its password, search for VPN credentials, or change network settings.
- After the first failure plausibly caused by a disconnected VPN (timeout, unreachable host, no route, or similar), stop remote work and ask the user to connect the VPN manually. Do not retry until the user says to continue. At most one additional diagnostic attempt is allowed when needed to distinguish network failure from authentication failure.

## Experiment workflow

1. Finish implementation and all relevant local checks first. Review the diff and determine the exact experiment command and expected result path.
2. Synchronize the intended Git revision to the remote repository reproducibly. Construct the exact remote command and invoke it through the command execution tool so the built-in approval UI prompts the user. Never run a destructive remote command such as `git reset --hard` without built-in approval for that exact command.
3. Submit GPU workloads only through the existing `submit_gpu` script; never run them directly on the SLURM login node. Prefer its non-interactive mode and specify the profile, resources, job name, workdir, and full experiment command explicitly. It accepts `--non-interactive`, `--profile`, `--partition`, `--account`, `--time`, `--mem`, `--cpus`, `--gpus`, `--job-name`, `--workdir`, mail options, repeated `--command`, `--command-file`, or an argv command after `--`. It submits with `sbatch --parsable` and prints the job ID. Invoke the exact remote submission command through the command execution tool so the built-in approval UI prompts the user.
4. When asked to check a job, use the simplest suitable approved command (`squeue`, `sacct`, `scontrol`, or a relevant log tail/cat). Do not poll repeatedly unless explicitly requested.
5. When results are ready, identify the useful files and avoid large checkpoints, caches, weights, or intermediate arrays unless needed. Invoke the exact transfer command through the command execution tool so the built-in approval UI prompts the user, preserve the relative path under remote `results/`, and analyze the downloaded files locally.

The `submit_gpu` profiles are `b200` (default: partition `p_b200_eng`, account `ug_lindenbaum`) and `uriofir` (partition `p_uriofir`, account `ug_uri_ofir`). Its resource defaults are time `01:00:00`, host RAM `32G`, 4 CPUs, and 1 GPU; do not depend on defaults silently when reproducibility benefits from explicit values.
