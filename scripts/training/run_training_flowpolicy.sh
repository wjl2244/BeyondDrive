export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

experiment_name=flowpolicy_beyonddrive  # replace with your experiment name
experiment_uid=$(date +%Y.%m.%d.%H.%M.%S)
output_dir=exp/flowpolicy_training/${experiment_name}/${experiment_uid}

lr=3e-4
batch_size=4  # 4 for training on 8 GPU
max_epochs=100

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    experiment_name=${experiment_name} \
    experiment_uid=${experiment_uid} \
    train_test_split=navtrain \
    trainer.params.num_nodes=1 \
    trainer.params.max_epochs=${max_epochs} \
    dataloader.params.batch_size=${batch_size} \
    agent=flowpolicy_agent \
    agent.lr=${lr} \
    agent.config.num_sampling_steps=5 \
    agent.config.num_sampling_proposals=1 \
    agent.config.cfg_scale=1.0 \
    agent.config.ckpt_path=${experiment_name} \
    cache_path=${NAVSIM_CACHE_ROOT}/traintest_v1_cache \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    output_dir=$output_dir \
    
