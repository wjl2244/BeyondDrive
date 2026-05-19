
export HYDRA_FULL_ERROR=1

NAVSIM_WORKSPACE="xxx/navsim_workspace"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive"
export NAVSIM_EXP_ROOT="${NAVSIM_WORKSPACE}/BeyondDrive/exp"
export OPENSCENE_DATA_ROOT="${NAVSIM_WORKSPACE}/dataset"
export NUPLAN_MAPS_ROOT="$OPENSCENE_DATA_ROOT/maps"
export NAVSIM_CACHE_ROOT="${NAVSIM_WORKSPACE}/cache"

TRAIN_TEST_SPLIT=navtest  # or navtrain

python navsim/planning/script/run_dataset_caching.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
experiment_name=debug \
cache_path=${NAVSIM_CACHE_ROOT}/traintest_v1_cache \
output_dir=exp/debug