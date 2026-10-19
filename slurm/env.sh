# source slurm/env.sh  — needs TG_ROOT, TG_CACHE, TG_RUNS, TG_SLURM_ACCOUNT, TG_VENV
: "${TG_ROOT:?}" "${TG_CACHE:?}" "${TG_RUNS:?}" "${TG_SLURM_ACCOUNT:?}" "${TG_VENV:?}"
eval "$("$TG_VENV/bin/python" - <<'PY'
import sys
sys.path.insert(0, ".")
from treeground.run import load_config
c = load_config("configs/paths.yaml")["slurm"]
print(f"export SBATCH_ACCOUNT={c['account']} SBATCH_PARTITION={c['partition']} SBATCH_QOS={c['qos']} "
      f"SBATCH_TIMELIMIT={c['time']}")
PY
)"
