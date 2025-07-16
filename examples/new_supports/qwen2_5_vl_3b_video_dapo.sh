#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
VIDEO_PATH=$1
cp verl/utils/vila_remote_code/* ${MODEL_PATH}

num_video_frames=32
tokens_per_frame=60
max_num_batched_tokens=$((num_video_frames*tokens_per_frame+8192))

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=LongVideo-Reason/longvideo-reason@train \
    data.val_files=LongVideo-Reason/longvideo-reason@validation \
    data.video_dir=$VIDEO_PATH \
    data.format_prompt=./examples/format_prompt/r1v.jinja \
    data.mini_rollout_batch_size=128 \
    worker.actor.padding_free=true \
    worker.actor.ulysses_size=4 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.num_video_frames=$num_video_frames \
    worker.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1v.py:compute_score \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \
    trainer.experiment_name=qwen2_5_vl_3b_video_dapo \
    trainer.n_gpus_per_node=8 \
    trainer.val_before_train=false \
    trainer.val_freq=-1

