#! /bin/bash
set -e 
prefix="https://github.com/vfonov/deep-qc/releases/download/v0/"

echo "Downloading minimal results, to run pretrained model 227MB in total..."


function download {
    set -e 
    for f in $@;do
        if [ ! -e ${f} ];then
        curl --location "${prefix}/${f}" -o ${f}
        fi
    done
}

download models_minimal_01.tar.xz

echo "Unpacking..."

mkdir -p results
tar xJf models_minimal_01.tar.xz
