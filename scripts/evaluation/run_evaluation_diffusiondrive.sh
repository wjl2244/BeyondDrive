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
output_dir=${NAVSIM_EXP_ROOT}/eval_diffusiondrive

python ${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_gpu_v1.py \
    agent=diffusiondrive_agent \
    dataloader.params.batch_size=32 \
    agent.checkpoint_path=${checkpoint_path} \
    trainer.params.precision=32 \
    output_dir=${output_dir} \
    cache_path=${NAVSIM_CACHE_ROOT}/trainval_v1_cache \
    metric_cache_path=${NAVSIM_CACHE_ROOT}/${split}_v1_metric_cache \
    train_test_split=${split}
