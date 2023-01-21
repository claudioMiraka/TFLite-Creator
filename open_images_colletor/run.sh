#!/bin/bash

echo 'Downloading...'

classes=()
while read line; do
    classes+=( "$line" )
done < classes.txt

python3 main.py downloader --classes "${classes[@]}" --type_csv all

echo 'Creating data set...'

python create_datasets.py

echo 'Done'

