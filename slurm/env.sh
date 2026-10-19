# Sourced by tg_env.sh: exports SBATCH_* defaults from configs/paths.yaml (slurm section).
if _tg_sbatch="$(cd "$TG_REPO" && python -c '
from treeground.run import load_config
c = load_config("configs/paths.yaml")["slurm"]
out = {"SBATCH_ACCOUNT": c["account"], "SBATCH_PARTITION": c["partition"], "SBATCH_TIMELIMIT": c["time"],
       "SBATCH_QOS": c.get("qos") or ""}
print(" ".join(f"export {k}={v};" for k, v in out.items() if v))
')"; then
    eval "$_tg_sbatch"
else
    echo "WARNING: could not read SLURM settings from configs/paths.yaml" >&2
fi
unset _tg_sbatch
