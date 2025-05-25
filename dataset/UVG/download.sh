#!/bin/bash

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
    # rm "$file"
done

mkdir -p videos_crop

for file in videos/*.yuv; do
    echo "Cropping $file"
    ffmpeg -pix_fmt yuv420p -s 1920x1080 -i "./videos/${file}.yuv" -vf crop=1920:1024:0:0 "./videos_crop/${file/1080/1024}.yuv"
done

python3 convert.py

cd CreateI

for crf in 20 23 26 29; do
    sh h265.sh $crf 1920 1024
    cp result.txt result_h265_crf_$crf.txt
done