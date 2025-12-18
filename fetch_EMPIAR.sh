#!/bin/bash

BASE_FTP="ftp://ftp.ebi.ac.uk/empiar/world_availability"

ID=$1
NUM_ID="${ID#EMPIAR-}"
mkdir -p "$ID"
cd "$ID" || exit
wget -r -np -nd -A "*.mrc" "${BASE_FTP}/${NUM_ID}/"
wget -r -np -nd -A "*.tiff" "${BASE_FTP}/${NUM_ID}/"
cd ..
