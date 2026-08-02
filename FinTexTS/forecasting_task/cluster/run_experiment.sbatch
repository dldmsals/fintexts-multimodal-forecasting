#!/bin/bash
#SBATCH -J fintexts
#SBATCH -o /home/eunmin/LG_AI/FinTexTS/logs/slurm_%x.%j.out
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# GPU 수동 지정 금지: Slurm이 --gres=gpu:1 로 할당한 GPU를 그대로 사용.
# CUDA_VISIBLE_DEVICES를 여기서 절대 하드코딩하지 않는다.

cd /home/eunmin/LG_AI/FinTexTS
python3 -u -m forecasting_task.run "$@"
