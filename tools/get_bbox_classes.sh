#!/bin/bash
# Get the number of bboxes saved for each class.
# Usage: /bin/bash get_bbox_classes.sh </path/to/txt/file> </path/to/labels>

pedestrians=0
vehicles=0

readarray -t a < "$1"

for txt in "${a[@]}"
do
	filename=$(basename "$txt")
	filename="${filename%.*}.txt"
    while read line; do
    	# Read each label line
		class=$(echo $line | head -n1 | awk '{print $1;}')
		if [ "$class" == 1 ]
		then
			pedestrians=$((pedestrians+1))
		elif [ "$class" == 0 ]
		then
			vehicles=$((vehicles+1))
		fi
	done < "$2"/"$filename"
done

echo "Vehicles: $vehicles"
echo "Pedestrians: $pedestrians"