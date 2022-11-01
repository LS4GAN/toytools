# Scripts

## Process proto-dune single phase data
- script: `pdsp-v0/process_raw.py`
- sample config file: `pdsp-v0/process_raw_config.yaml`
- command: (in side pdsp-v0 folder)
  > `python process_raw.py --config_fname=process_raw_config.yaml --num_workers=32`

More help could be found in the python script.

## Center cut, train-test split, and tiling
After the raw data is processed, we can either run the 3 scripts in `toyzero/`
to get center cut, train-test split and tiles in the following order:
1. `center_crop`
1. `train_test_split`
1. `tile_crop`

Or we can run
> `bash combined_commands.sh`

to run them all at once.
