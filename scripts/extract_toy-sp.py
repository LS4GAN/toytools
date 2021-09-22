#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
from pathlib import Path
import tarfile
import os
import shutil
import sys
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt



# Extraction functions
def extract_files(tar, member_folder, file_extension, save_folder='.'):
	"""
	Extract all files with <file_extension> from folder <member_folder> of a tarfile object <tar>
	Input:
		- tar (tarfile instance): a tarfile instance to extract from.
		- member_folder (str): a folder the contains the files you need. Must be a member in <tar>.
		- file_extension (str): file extension for the files you need.
		- save_folder (str): the folder to extract to.
	"""
	
	Path(save_folder).mkdir(exist_ok=True, parents=True)	
	# locate all the files in the member_folder that have file_extension as extension.
	subdir_and_files = [
		tarinfo for tarinfo in tar.getmembers()
		if tarinfo.name.startswith(member_folder) and 
		tarinfo.name.endswith(file_extension)
	]
	
	print(f'\tsave {len(subdir_and_files)} {file_extension} files to {save_folder}')
	
	# extract files and  move them to the save_folder
	tar.extractall(members=subdir_and_files, path=save_folder)
	fnames = Path(f'{save_folder}/{member_folder}').glob(f'*.{file_extension}')
	for fname in fnames:
		shutil.move(fname, save_folder)
	
	# Remove the chain of parent folders (now empty)
	path_base = os.path.normpath(member_folder).split(os.sep)[0]
	shutil.rmtree(f'{save_folder}/{path_base}')

def get_seed(fname):
	return fname.split('-')[-1].split('_')[0]

def extract_toy_sp(tarfname, folder_base):
	"""
	extract all the npz files (both fake and real) from a tar file named <tarfname>
	Input: 
		- tarfname (str): the .tar file name.
		- folder_base (str): the folder under where the seed-<seed> folders are located
	"""
	seed = get_seed(Path(tarfname).stem)
	print(f'seed = {seed}')
	save_folder = Path(f'{folder_base}/seed-{seed}')
	if save_folder.exists():
		shutil.rmtree(save_folder)
	
	with tarfile.open(tarfname, 'r') as tar:
		try:
			fake = [tarinfo.name for tarinfo in tar.getmembers() if tarinfo.name.endswith('fake-fake')][0]
			real = [tarinfo.name for tarinfo in tar.getmembers() if tarinfo.name.endswith('real-fake')][0]
			extract_files(tar, fake, 'npz', save_folder/'fake')
			extract_files(tar, real, 'npz', save_folder/'real')
			print()
		except:
			print(f'There is something wrong with the tar file {tarfname}')


def parse_cmdargs():
	parser = argparse.ArgumentParser("Extract toy-sp (toy1) data Brett generated")

	parser.add_argument(
		'--data_path',
		'-d',
		help	= 'The loacation of the tar files',
		dest	= 'data_path',
		type	= str,
	)

	parser.add_argument(
		'--seed_path',
		'-s',
		help	= 'The location for the seed folders. Images generated with seed=<seed> are contained in <seed_path>/seed-<seed>.',
		dest	= 'seed_path',
		type	= str,
		default = None
	)

	parser.add_argument(
		'--merged_path',
		'-m',
		dest	= 'merged_path',
		help	= 'The location of the merged dataset. The folder will contain two subfolders: fake and real.',
		type	= str
	)

	parser.add_argument(
		'--visualize',
		'-v',
		help	= 'whether to visualize a few pairs of fake and real images',
		dest	= 'visualize',
		default = False,
		type	= bool,
	)

	return parser.parse_args()



if __name__ == '__main__':
	cmdargs 	= parse_cmdargs()
	data_path 	= cmdargs.data_path
	seed_path 	= cmdargs.seed_path
	merged_path	= cmdargs.merged_path
	visualize 	= cmdargs.visualize

	# extract the image npz files from each tarfile and save them to folder according to their seed
	# Each seed-<seed> folder looks like
	# seed-<seed>/
	#	 - fake/
	#	 - real/

	print('Extracting:')
	assert Path(data_path).exists(), f"{data_path} does not exist"

	tarfnames = list(Path(data_path).glob('*.tar'))
	assert len(tarfnames) > 0, f"{data_path} does not contain any tar files"

	if not seed_path:
		seed_path = '/tmp/LS4GAN/toy-sp/seedwise/'
		print('\033[96m' + f'Find seedwise data at {seed_path}' + '\033[0m')
		
	if not Path(seed_path).exists():
		Path(seed_path).mkdir(exist_ok=True, parents=True)

	for tarfname in tarfnames:
		extract_toy_sp(tarfname, seed_path)


	# Merge
	# Modified the image filenames and then save copies to `<merged_path>`.
	# The merged folder has two subfolders `fake` and `real`.
	merged_folder = Path(merged_path)
	merged_folder.mkdir(exist_ok=True, parents=True)
	merged_fake = merged_folder/'fake'
	merged_real = merged_folder/'real'
	merged_fake.mkdir(exist_ok=True, parents=True)
	merged_real.mkdir(exist_ok=True, parents=True)

	print('Merging:')
	for folder in Path(seed_path).glob('seed-*'):
		seed = folder.stem.split('-')[-1]
		print(f'seed = {seed}')

		npz_fnames = list((folder/'fake').glob('*npz'))
		print(f'\tcopy {len(npz_fnames)} to the {merged_fake} folder')
		for npz_fname in sorted(npz_fnames):
			npz_fname_new = str(npz_fname.name).replace('gauss', f'{seed}-')
			shutil.copy(npz_fname, merged_fake/npz_fname_new)

		npz_fnames = list((folder/'real').glob('*npz'))
		print(f'\tcopy {len(npz_fnames)} to the {merged_real} folder')
		for npz_fname in sorted(npz_fnames):
			npz_fname_new = str(npz_fname.name).replace('gauss', f'{seed}-')
			shutil.copy(npz_fname, merged_real/npz_fname_new)
		print()


	# Visualization
	if not visualize:
		exit(0)

	print('Generating sample plots')
	def load_image(fname):
		with np.load(fname) as f:
			return f[f.files[0]]

	fnames_fake = sorted((merged_fake).glob('*npz'))
	fnames_real = sorted((merged_real).glob('*npz'))
	num_samples = 5
	image_save_folder = Path(f'/tmp/LS4GAN/toy-sp/sample_plots')
	if image_save_folder.exists():
		shutil.rmtree(image_save_folder)
	image_save_folder.mkdir(exist_ok=True, parents=True)
	print('\033[96m' + f'Find sample images at {image_save_folder}' + '\033[0m')
	indices = np.random.choice(range(len(fnames_fake)), num_samples, replace=False)
	
	for idx in tqdm(indices):
	    image_fname = image_save_folder/f'sample_{idx}.png'
	    fname_fake = fnames_fake[idx]
	    fname_real = fnames_real[idx]
	
	    image_fake = load_image(fname_fake)
	    image_real = load_image(fname_real)
	
	    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
	    axes[0].pcolormesh(image_fake)
	    axes[1].pcolormesh(image_real)
	    axes[2].pcolormesh(image_fake - image_real)
	    plt.savefig(image_fname, dpi=200, bbox_inches='tight')
