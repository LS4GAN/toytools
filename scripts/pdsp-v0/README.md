# Process proto-dune single phase data version 0
- script: `process.py`
- sample config file: `config.yaml`
- command:
  > `python process_raw.py --config_fname=config.yaml --num_workers=32`

More help could be found in the python script.

After processing the data, modify and run `./combined_command.sh` to make a dataset.
