# requires three commandline args - 
# 1. CPU cores required
# 2. RAM in GB
# 3. GPU required
# 4. Number of Hours

echo
echo "Getting Session with $1 core(s), $2 GB ram, $3 GPU(s) for $4 hours..."
srun -c$1 --mem="$2GB" --gres=gpu:rtx8000:$3 -t$4:00:00 --pty /bin/bash