module purge
module load gcc/10.2.0
module load cmake/3.22.2 
module load ffmpeg/4.2.4 
module load cuda/11.3.1
module load anaconda3/2020.07 
echo "Modules Loaded"
eval "$(conda shell.bash hook)"