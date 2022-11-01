#!/bin/bash

# Flag arguments
helpstring="
#============================= Central cut, split, and tiling =================================#
script usage: $(basename $0) [-s scriptroot] [-m min_signal] [-d dataroot] [-p sig_proc_methods]

[dataroot]: the root of the wire plane-separated full frames
            Dataroot should have the following structure:

            dataroot
            ├── sig_proc_method_1
            │   ├── fake
            │   └── real
            └── sig_proc_method_2
                ├── fake
                └── real

[scriptroot]: root to the scripts of center cut, split, and tiling
              (default = ./toyzero)

[min_signal]: the number of nonzero pixels for a tile to be kept
              (default = -1)

[sig_proc_methods]: signal processing methods
                    (default = \"gauss orig\")
                    Input as -p \"method_1 method_2\"
                    with space-separated items in a pair of quotation marks.
#==============================================================================================#
"

scriptroot=../toyzero
min_signal=-1
sig_proc_methods="gauss orig"

while getopts 's:m:d:p:h' OPTION; do
    case "$OPTION" in
        s) scriptroot="$OPTARG" ;;
        m) min_signal="$OPTARG" ;;
        d) dataroot="$OPTARG" ;;
        p) sig_proc_methods="$OPTARG" ;;
        h) echo "$helpstring"; exit 1 ;; # help
        ?) echo "$helpstring"; exit 1 ;; # default
    esac
done

if ! [[ -v dataroot ]];
then
    echo "dataroot is not given"
    exit 1
fi

# Check the arguments.
# If there is anything wrong, user can exit.
echo script root = "$scriptroot"
echo min signal = "$min_signal"
echo data root = "$dataroot"
echo "sigal processing method(s):"
for sig_proc_method in $sig_proc_methods
do
    echo "  - $sig_proc_method"
done
read -n 1 -s -r -p "Press any key to continue or control+c to exit"


# Process data
for sig_proc_method in $sig_proc_methods
do
    # We first generate a center crop of size 768x5888 from 800 x 6000 sized full frame
    python ${scriptroot}/center_crop ${dataroot}/${sig_proc_method}/ ${dataroot}/${sig_proc_method}_center_crop --plane U -s 768x5888
    # split the central-cuts into train and test
    python ${scriptroot}/train_test_split ${dataroot}/${sig_proc_method}_center_crop ${dataroot}/${sig_proc_method}_split
    # cut the central-cuts into 256 by 256 crops with a given min_signal
    python ${scriptroot}/tile_crop ${dataroot}/${sig_proc_method}_split ${dataroot}/${sig_proc_method}_tile_${min_signal} -s 256x256 --min-signal ${min_signal}
done
