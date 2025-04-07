#!/bin/bash
#SBATCH --job-name=hg_sateer        # Job name
#SBATCH --output=output.txt           # Standard output file
#SBATCH --error=error.txt             # Standard error file
#SBATCH --partition=department_only    # Partition or queue name
#SBATCH --gpus=1
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --time=72:00:00                # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --mail-user=lanzino@di.uniroma1.it    # Email address for notifications

# creates a new environment
conda create -y --name=test_env
conda activate test_env
pip install -r requirements.txt
bash run_all_tests.sh configs/hg_.*sateer.*