#!/usr/bin/env bash
################################################################
# Usage, bash extract_all_rosbags_data.sh DIRECTORY, this will extract the data from all the rosbags in DIRECTORY
# This will only extract in the files inside the folder and will not look for subfolders, in case you want to extract rosbags inside subfolders, please modify the maxdepth parameter or erase it
####################################################################
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Temporarily setting the internal field seperator (IFS) to the newline character.
IFS=$'\n';

# Recursively loop through all bag files in the specified directory
BAG_DIRECTORY=$1
#Command -maxdepth 1 makes sure it only searches in the folder and not subfolders
for bag in $(find ${BAG_DIRECTORY} -maxdepth 1 -name '*.bag'); 
do
	echo "Processing bag file ${bag}"
	INPUT_DATASET_FILE="${bag}"

	rosrun ml_rosbag_extractor single_rosbag_extractor.py ${bag}
    # rosrun ml_rosbag_extractor waveform_converter.py ${bag}

	echo "==============================================================================="
	sleep 1
done
