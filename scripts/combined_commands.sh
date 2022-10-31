#!/bin/bash

min_signal=${1:--1}
dataroot=/data/datasets/LS4GAN_yuhw/full_images
scriptroot=~/PROJs/LS4GAN/toytools/scripts/toyzero
echo min signal = $min_signal
read -n 1 -s -r -p "Press any key to continue"

for sigproc in gauss orig
do
    echo signal processing = $sigproc
    # We first generate a center crop of size 768x5888 from 800 x 6000 sized full frame
    cmd_center_crop="python ${scriptroot}/center_crop ${dataroot}/${sigproc}/ ${dataroot}/${sigproc}_center_crop --plane U -s 768x5888"
    # split the central-cuts into train and test
    cmd_split="python ${scriptroot}/train_test_split ${dataroot}/${sigproc}_center_crop ${dataroot}/${sigproc}_split"
    # cut the central-cuts into 256 by 256 crops with a given min_signal
    cmd_tile="python ${scriptroot}/tile_crop ${dataroot}/${sigproc}_split ${dataroot}/${sigproc}_tile_${min_signal} -s 256x256 --min-signal ${min_signal}"

    $cmd_center_crop
    $cmd_split
    $cmd_tile
done
