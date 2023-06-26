#!/bin/sh

echo "Running Inference on CPU"
python3 ./main.py -i ./resources/Pedestrain_Detect_2_1_1.mp4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://192.168.0.10:3004/fac.ffm
