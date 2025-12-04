#!/bin/bash
#SBATCH --job-name=converthits                  # Job name
#SBATCH --output=logs/converthits.%j.out        # Stdout
#SBATCH --error=logs/converthits.%j.err         # Stderr
#SBATCH --cpus-per-task=1                       # Ncpus
#SBATCH --time=02:00:00                         # Runtime    
#SBATCH --mem=32GB                              # Memory per node

source /lustre/nyx/panda/markhoff/geant

export PATH=/lustre/nyx/panda/markhoff/local/python3.11/bin:$PATH

python3 convert_film.py ../../../data/iahits.root
