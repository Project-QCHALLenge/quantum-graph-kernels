#!/bin/bash
#
# Name of the job
#SBATCH --job-name=QGK_Wheel_Watts-2

# Send status information to this email address
#SBATCH --mail-user=Sabrina.Egger@campus.lmu.de
#
# Send an e-mail when the job has finished or failed
#SBATCH --mail-type=END,FAIL
#
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=ALL
#SBATCH --partition=All
#SBATCH --time=7-00:00:00

python3 main.py
