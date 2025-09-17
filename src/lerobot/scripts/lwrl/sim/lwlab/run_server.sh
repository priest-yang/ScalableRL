#!/bin/bash
task_config=lerobot_liftobj_visual
env_gpu=0
policy_gpu=0

if [ "$env_gpu" -eq "$policy_gpu" ]; then
    export CUDA_VISIBLE_DEVICES=${env_gpu}
    export ENV_GPU=0
    export POLICY_GPU=0
else
    export CUDA_VISIBLE_DEVICES="${env_gpu},${policy_gpu}"
    export ENV_GPU=0
    export POLICY_GPU=1
fi

export LW_API_ENDPOINT="https://api-dev.lightwheel.net"
python ./env_serve.py \
    --task_config="$task_config" \
    # --headless \
