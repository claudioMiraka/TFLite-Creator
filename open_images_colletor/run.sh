#!/bin/bash

echo 'Downloading...'

classes=()
while read line; do
    classes+=( "$line" )
done < classes.txt

echo "${classes[@]}"

python3 main.py downloader --classes "${classes[@]}" --type_csv all

echo 'Creating data set...'

python3 create_datasets.py --classes "${classes[@]}" 

echo 'Done'

