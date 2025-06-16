#!/bin/bash
#SBATCH --partition=Apus
#SBATCH -J Au-SH
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=96
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --mem=96G     

module load vasp/6.4.3
echo "Running on node: $(hostname)"
echo "Job Dir : $SLURM_SUBMIT_DIR"
echo "Job ID/Name : $SLURM_JOBID / $SLURM_JOB_NAME"
echo "Start Time : $(date)"

start_time=$(date +%s)
mpirun -np 24 python gp_neb.py > log_gp_neb.dat
end_time=$(date +%s)
run_time=$((end_time - start_time))
echo "Total runtime: $run_time seconds"
