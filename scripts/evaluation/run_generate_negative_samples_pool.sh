export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"


split=navtrain
num_sampling_steps=5
num_sampling_proposals=64
cfg_scale=0.5
noise_scale=0.4
checkpoint_path=path/to/checkpoint.ckpt
output_dir=path/to/output_dir/${num_sampling_steps}_proposal${num_sampling_proposals}_cfg${cfg_scale}_noise${noise_scale}


python navsim/planning/script/run_pdm_score_gpu_v1_proposals.py \
    cache_path=${NAVSIM_CACHE_ROOT}/trainval_v1_cache \
    metric_cache_path=${NAVSIM_CACHE_ROOT}/${split}_v1_metric_cache \
    train_test_split=${split} \
    agent=flowpolicy_agent \
    agent.checkpoint_path=${checkpoint_path} \
    agent.config.num_sampling_steps=${num_sampling_steps} \
    agent.config.num_sampling_proposals=${num_sampling_proposals} \
    agent.config.noise_scale=${noise_scale} \
    agent.config.cfg_scale=${cfg_scale} \
    output_dir=${output_dir} \
    trainer.params.precision=32 \
    worker.threads_per_node=92 \
    dataloader.params.batch_size=16
