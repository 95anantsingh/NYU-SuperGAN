echo
echo "Getting Session..."
srun -c20 --mem=32GB --gres=gpu:1 --gres=gpu:rtx8000:1 -t10:00:00 --pty /bin/bash

srun -c1 --mem=1GB --gres=gpu:1 -t1:00:00 --pty /bin/bash