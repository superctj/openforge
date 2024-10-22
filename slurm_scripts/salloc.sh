salloc --account=jag0 --partition=largemem --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=128g

salloc --account=jag98 --partition=gpu --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=128g --time=05:00:00

salloc --account=jag0 --partition=spgpu --gres=gpu:1 --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=64g --time=4:00:00