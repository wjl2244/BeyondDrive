export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

split=navtest
checkpoint_path=path/to/checkpoint.ckpt
output_dir=path/to/output_dir

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v1.py \
    cache_path=${NAVSIM_CACHE_ROOT}/traintest_v1_cache \
    metric_cache_path=${NAVSIM_CACHE_ROOT}/${split}_v1_metric_cache \
    train_test_split=${split} \
    agent=transfuser_agent \
    agent.checkpoint_path=${checkpoint_path} \
    output_dir=${output_dir} \
    trainer.params.precision=32 \
    dataloader.params.batch_size=32 \
    agent.config.use_delta_traj=True \
    agent.config.latent=True \
    
