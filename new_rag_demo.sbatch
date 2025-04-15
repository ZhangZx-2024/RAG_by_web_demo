#!/bin/bash
#SBATCH --job-name=rag_demo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=msccsit2024
#SBATCH --gres=gpu:1

module purge
module load Anaconda3/2023.09-0
source ~/myenv/bin/activate

cd ~/nlp/RAG_by_web_demo
python fix_script.py