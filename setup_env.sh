module purge
echo "Loading modules..."
module load python/3.12.11-fasrc02
module load cuda/12.9.1-fasrc01
module load cudnn/9.10.2.21_cuda12-fasrc01
export UV_PYTHON_PREFERENCE=only-system

echo "module Loaded."
