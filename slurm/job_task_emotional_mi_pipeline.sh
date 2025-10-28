#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./test_job.out.%j
#SBATCH -e ./test_job.err.%j

# Initial working directory:
#SBATCH -D ./

# Job Name:
#SBATCH -J my_test_job

# Set up the job for a single task and single GPU:
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1

# Resource constraints
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

# job's runtime:
#SBATCH --time=02:00:00


# Activate the Miniconda virtual environment:
source /u/kamu/miniforge3/bin/activate transformersenv

# Display the current time:
echo "Job started at: $(date)"

#Navigate to the correct directory to run the script as a module
cd ~/EmotionsMechInt

#Display the completion time
echo "Job finished at: $(date)"

