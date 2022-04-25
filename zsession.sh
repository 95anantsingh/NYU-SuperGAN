# requires three commandline args - 
# 1. CPU cores required
# 2. RAM in GB
# 3. GPU required
# 4. Number of Hours

echo
echo "Getting Session with $1 Cores, $2 GB Ram, $3 GPUs for $4 Hours..."
srun -c$1 --mem="$2GB" --gres=gpu:rtx8000:$3 -t$4:00:00 --pty /bin/bash