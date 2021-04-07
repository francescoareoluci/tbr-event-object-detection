#!/bin/bash
# Change the path of images in train, valid and test txt files.
# Can be used when the encoded dataset is moved in another path.
# Usage: /bin/bash change_dataset_path.sh <src_dataset_root_folder> <dst_dataset_root_folder>

src_folder="$1"
new_path="$2"

src_relative_path="$src_folder"/data/custom
dest_relative_path="$new_path"/data/custom/images
train_txt_file="$src_relative_path"/train.txt
valid_txt_file="$src_relative_path"/valid.txt
test_txt_file="$src_relative_path"/test.txt

check_txt_files() {
	local txt_file_path="$1"
	if [ ! -f "$txt_file_path" ]
	then
		echo "File $txt_file_path does not exists, exiting..."
		exit 1
	fi
}

change_paths() {
	local tmp="$1"/swp.txt
	local src="$2"

	touch "$tmp"

	while read line; do
		filename=$(basename "$line")
		new_line="$dest_relative_path"/"$filename"
		echo "$new_line" >> "$tmp"
	done < "$src"

	mv "$tmp" "$src"
}

check_txt_files "$train_txt_file"
check_txt_files "$valid_txt_file"
check_txt_files "$test_txt_file"
change_paths "$src_relative_path" "$train_txt_file"
change_paths "$src_relative_path" "$valid_txt_file"
change_paths "$src_relative_path" "$test_txt_file"
