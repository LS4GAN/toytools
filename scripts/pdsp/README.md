# Process proto-dune single phase data
- script: `pdsp/process.py`
- sample config file: `pdsp/config.yaml`
- command: (in side pdsp folder)
  > `python process_raw.py --config_fname=config.yaml --num_workers=32`

More help could be found in the python script.

After processing the data, modify and run
`/script/combined_command.sh` to make a dataset.
