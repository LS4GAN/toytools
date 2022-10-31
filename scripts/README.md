# Process raw (simulation) data provided by Haiwang or other physicists
- script: `process_raw.py`
- config: `process_raw_config.yaml`
- command: 
  > `python process_raw.py --config_fname=process_raw_config.yaml --num_workers=32`

More help could be found in the python script.

# Center cut, train-test split, and tiling
After the raw data is processed, we can either run the 3 scripts in `toyzero/`
to get center cut, train-test split and tiles.
Or we can run 
> `bash combined_commands.sh` 

to run them all at once.
