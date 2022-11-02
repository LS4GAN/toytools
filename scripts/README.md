# Scripts

## [`toyzero`](./toyzero): Center crop, train-test split, and tiling
With wire plane-separated full frames (U, V: 800 x 6000, W: 960 x 6000) as input, 
the three scripts in `./toyzero` does the following:
1. Center_crop: crop a the center rectangle from the frames
1. train_test_split: split the center crops into train, test, valid
1. tile_crop: extract non-overlapping tiles from the center crops.

*NOTE*: the scripts has to be run in the order above.
More help can be found in the scripts.

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
