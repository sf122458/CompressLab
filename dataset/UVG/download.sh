#!/bin/bash

# A shell script to process the UVG dataset.
# **Attention**: check if you have installed `ffmpeg` and `7z` before running this script
# Please use `sudo apt-get install ffmpeg p7zip-full` to install them.

# The final structure of the dataset will be:
# - images/
#   - Beauty
#       - H265L20/
#           - im0001.png
#           ...
#       - H265L23/
#       - H265L26/
#       - H265L29/
#       - im001.png
#       ...
#   - Bosphorus
#   - HoneyBee
#   - Jockey
#   - ReadySetGo
#   - ShakeNDry
#   - YachtRide

mkdir videos

download_links=(
    "https://ultravideo.fi/video/Beauty_1920x1080_120fps_420_8bit_YUV_RAW.7z"
    "https://ultravideo.fi/video/Bosphorus_1920x1080_120fps_420_8bit_YUV_RAW.7z"
    "https://ultravideo.fi/video/HoneyBee_1920x1080_120fps_420_8bit_YUV_RAW.7z" 
    "https://ultravideo.fi/video/Jockey_1920x1080_120fps_420_8bit_YUV_RAW.7z" 
    "https://ultravideo.fi/video/ReadySetGo_1920x1080_120fps_420_8bit_YUV_RAW.7z" 
    "https://ultravideo.fi/video/ShakeNDry_1920x1080_120fps_420_8bit_YUV_RAW.7z" 
    "https://ultravideo.fi/video/YachtRide_1920x1080_120fps_420_8bit_YUV_RAW.7z"
)

for link in "${download_links[@]}"; do
    echo "Downloading $link"
    wget -c "$link" -P videos/
done

for file in videos/*.7z; do
    echo "Extracting $file"
    7z x "$file" -ovideos/
done

rm videos/*.txt
rm videos/*.7z

mkdir -p videos_crop

for path in videos/*.yuv; do
    file=$(basename "$path")
    echo "Cropping $file"
    ffmpeg -pix_fmt yuv420p -s 1920x1080 -i "./videos/${file}" -vf crop=1920:1024:0:0 "./videos_crop/${file/1080/1024}"
done

rm -rf videos

for src in videos_crop/*.yuv; do
    filename=$(basename "$src" .yuv)
    prefix="${filename%%_*}"
    dst="images/$prefix"
    mkdir -p $dst
    ffmpeg -y -pix_fmt yuv420p -s 1920x1024 -i $src $dst/im%03d.png
done

cd CreateI

for crf in 20 23 26 29; do
    sh h265.sh $crf 1920 1024
    cp result.txt result_h265_crf_$crf.txt
done

rm -rf ../videos_crop