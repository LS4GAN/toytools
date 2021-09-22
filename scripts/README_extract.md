# Get `toy-sp` (`toy1`) data:
Toy1 data can be downloaded from https://www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/.
1. To download the both image `.npz` files and meta data:
    > `wget -r --no-parent https://www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/`
1. To download only the image `.npz` files:
    > `wget -r --no-parent https://www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/job_output_dir`
1. To get images generate with one particular seed (example):
    >  `wget -r --no-parent https://www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/job_output_dir/toyzero-100-10-17698759_1048280_804.tar`

When the downloading is done, we can find the tar files that contains the images in the following folder:
> `www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/job_output_dir`

# Extract `toy-sp` (`toy1`) data:
We use `extract_toy-sp.py` to extract the tar files and get a merged dataset contains all images (generated with different seeds) 

Usages:
1. `python extract_toy-sp.py -d www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/job_output_dir/ -s seedwise -m merged -v True`;
    Note: Sampled pairs of images will be saved to `flora:/tmp/LS4GAN/toy-sp/sample_plots/`
1. `python extract_toy-sp.py -d www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/job_output_dir/ -m merged -v True`;
    Note: If seedwise folder is not specified, the folders for each seed will be saved at `flora:/tmp/LS4GAN/toy-sp/seedwise/`.
1. `python extract_toy-sp.py -d www.phy.bnl.gov/~bviren/tmp/ls4gan/data/toy1/job_output_dir/ -m merged`;
    Note: Do not plot sampled pairs. 

# Bulk data downloading and pre-generated window csvs:
Please find in `flora:/data/LS4GAN` folder of the `flora` machine.
1. **Merged data**: A gathering of all toy1 image `.npz` files Brett generated.
1. **Seed-wise data**: The seed-wise data contains the same set of toy1 images as the merged data, but grouped and saved by the seed.  
    
    | file | md5sum |
    | ------------------------------ | -------------------------------- |
    | 2021-09-01_toy-sp_merged.tgz   | bfbd560f65955d66d5d174f0f410eee8 |
    | 2021-09-01_toy-sp_seedwise.tgz | b5dde35f59283313998f5ecea51b3340 |
1. **Pre-generated of windows csvs**: in the `flora:/data/LS4GAN/toy-sp_merged_windows`.
    - Shared parameters for `toytools/scripts/preprocess`: 
        - `--plane`: U;
        - `-n` (numbers of windows per image): 4;
    - Shared parameters for `toytools/scripts/train_test_split`:
        - `--test-size`: 4000;
    - For window size `128x128`:
        - `--min-signal`: 300, 500;
    - For window size `512x512`:
        - `--min-signal`: 500, 1000, 2000;
    - Window file naming convention:
        > `ms<min-signal>-<plane>-<shape>.csv`;
        > 
        > `ms<min-signal>-<plane>-<shape>_train.csv`;
        > 
        > `ms<min-signal>-<plane>-<shape>_test.csv` 
