# Scripts

## [`pdsp-v0`](./pdsp-v0): Process proto-dune single phase data, version 0
### Step 1: process the raw output:
  - script: `pdsp-v0/process.py`
  - sample config file: `pdsp-v0/config.yaml`
  - usage example: (inside `pdsp-v0`)
    > `python process_raw.py --config_fname=config.yaml --num_workers=32`
### Step 2: make a dataset
After the raw data is processed, we can either run the three scripts in `toyzero/`
to get center crop, train-test split, and tiles (in this order). Or we can run (inside `pdsp-v0`)
> `bash combined_commands.sh -d [datapath]`

to run them all at once.

More help could be found in the python and bash script.

## [`toyzero`](./toyzero): Center crop, train-test split, and tiling
