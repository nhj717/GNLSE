#!/bin/bash
# Job name. This name will be shown in the queue overview
#SBATCH --job-name=comsol
# Output and error files. They will be written in the current working directory.
#SBATCH --output=logs/comsol_outputs.out
#SBATCH --error=logs/comsol_error.err
# Time the job will run. After this time the job will be forced-stopped.
#SBATCH --time=2-00:00:00
# We select the partition
#SBATCH --partition=delta-new
#One node per parameter value job (embarrassingly parallel)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#Number of CPU (Cores) for this task
#SBATCH --cpus-per-task=20

#SBATCH --array=0-9%5
set -euo pipefail
pwd; hostname; date
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# We load the comsol module asking for version 6.3
module load comsol/6.3

#Define model and outer parameter values
MODEL="twisted_5ring_sweep/cmap_test"
P1_TAG="d_ratio"

#sweep definition
N=10
P1_MIN=1.0
P1_MAX=1.9
i=$SLURM_ARRAY_TASK_ID
P1=$(python3 - <<PY
import math
N=${N}
pmin=${P1_MIN}
pmax=${P1_MAX}
i=${i}
val = pmin + (pmax-pmin)*i/(N-1)
print(f"{val:.6g}")
PY
)

INPUTFILE="input/${MODEL}.mph"
TMPBASE="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
MPH_TMP="${TMPBASE}/${P1_TAG}_${P1}.mph"
OUTPUTDIR="output/${MODEL}"
OUTPUTFILE="${OUTPUTDIR}/${P1_TAG}_${P1}.mph"
LOGDIR="logs/${MODEL}"
LOG="${LOGDIR}/comsol_${P1_TAG}_${P1}.log"
mkdir -p "$TMPBASE"
mkdir -p ~/${OUTPUTDIR}
mkdir -p ~/${LOGDIR}
comsol batch -np ${SLURM_CPUS_PER_TASK} -inputfile ${INPUTFILE} -outputfile ${OUTPUTFILE} -study std1 -pname ${P1_TAG} -plist ${P1} -batchlog ${LOG}

module load matlab/R2025b

# final HDF5 output in home (small enough)
H5OUT="${OUTPUTDIR}/${P1_TAG}_${P1}.h5"
# Safety checks
test -f "${OUTMPH}" || { echo "Missing solved model: ${OUTMPH}"; exit 1; }
test -f "input/ModeTrack.m" || { echo "Missing MATLAB script: input/ModeTrack.m"; exit 1; }

# Stagger starts: task 0 starts immediately, task 1 after 30s, task 2 after 60s, ...
DELAY=$((60* SLURM_ARRAY_TASK_ID))
echo "Sleeping ${DELAY}s to stagger mphserver startup..."
sleep "${DELAY}"

# ---- avoid ~/.comsol workspace issues + give each task its own dirs ----
PORTFILE="${TMPBASE}/port.txt"
SVRLOG="${LOGDIR}/mph_${P1_TAG}_${P1}.log"

export COMSOL_USER_HOME="${TMPBASE}/prefs"
export COMSOL_TMPDIR="${TMPBASE}/comsol_tmp"
export TMPDIR="${COMSOL_TMPDIR}"
mkdir -p "${COMSOL_USER_HOME}" "${COMSOL_TMPDIR}"

cleanup(){ [[ -n "${SVRPID:-}" ]] && kill "${SVRPID}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Start COMSOL server
comsol mphserver -silent -login never -port 0 -portfile "${PORTFILE}" -np "${SLURM_CPUS_PER_TASK}" \
  > "${SVRLOG}" 2>&1 &
SVRPID=$!

# Wait for portfile to appear (up to ~60 s)
for i in {1..45}; do
  [[ -s "${PORTFILE}" ]] && break
  sleep 1
done
if [[ ! -s "${PORTFILE}" ]]; then
  echo "Portfile not created: ${PORTFILE}"
  echo "---- mphserver log tail ----"
  tail -n 120 "${SVRLOG}" || true
  kill "${SVRPID}" 2>/dev/null || true
  exit 1
fi

PORT=$(cat "${PORTFILE}")
SUBMITDIR="${SLURM_SUBMIT_DIR:-$PWD}"

#eport variables for the ModeTrackRunner.m file which will run the actual ModeTrack.m file
export SUBMITDIR OUTPUTFILE h5OUT P1

# ---- run MATLAB headless, connect with mphstart, run your script ----
START_TIME=$(date +%s)
cd /software/comsol/6.3/mli
matlab -nodesktop -nosplash -r "mphstart('localhost',${PORT}); run('${HOME}/input/ModeTrackRunner.m'); exit"

# ---- stop server ----
kill "${SVRPID}" 2>/dev/null || true

# ---- verify output ----
for i in {1..10}; do
  if [[ -s "$HOME/${h5OUT}" ]]; then
   echo "TXT successfully created: ${h5OUT}"
   END_TIME=$(date +%s)
   ELAPSED=$((END_TIME - START_TIME))
   H=$((ELAPSED/3600))
   M=$(((ELAPSED%3600)/60))
   S=$((ELAPSED%60))
   printf "Total evaluation time of %s_%s is %02d:%02d:%02d (hh:mm:ss)\n"\
    "${P1_TAG}" "${P1}" "$H" "$M" "$S"
   exit 0
  fi
  sleep 1
done

echo "ERROR: TXT  not created: ${h5OUT}"
exit 1

