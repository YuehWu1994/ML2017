#!/bin/bash
#python cnnSample.py relu 2 512 100 200 sgd 1 4 3 0.35 0.7 10 32 64 0    best
# original
#python cnnSample.py relu 3 512 100 200 sgd 1 4 3 0.35 0.7 10 32 64 0 0 3
# 3layerConv
#python cnnSample.py relu 3 512 100 200 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3
#python cnnSample.py relu 3 512 100 200 sgd 1 4 3 0.35 0.7 10 32 64 0 128 3
#python cnnSample.py relu 2 512 100 200 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3
#python cnnSample.py relu 2 512 100 200 sgd 1 4 3 0.35 0.7 10 32 64 0 128 3
#python cnnSample.py relu 2 512 100 200 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 best
#python cnnSample.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3
#python cnnSample.py relu 2 512 100 300 sgd 1 4 3 0.35 0.7 10 32 64 0 128 3
#python cnnSample.py relu 2 512 100 400 sgd 1 4 3 0.35 0.7 10 32 64 0 128 3 0 3
#python cnnSample.py relu 2 512 100 300 sgd 1 4 3 0.35 0.7 10 50 100 0 200 3 400 3
#python cnnSample.py relu 2 512 100 400 sgd 1 3 3 0.35 0.7 10 50 100 1 200 3 400 3
#python cnnSample.py relu 2 512 100 400 sgd 1 5 4 0.35 0.7 10 50 100 0 200 3 400 3
#python cnnSample.py relu 2 512 100 500 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3
#python cnnSample.py relu 2 512 100 400 sgd 1 3 3 0.35 0.7 10 64 128 0 256 3 512 3
#python cnnSample.py relu 2 512 100 400 sgd 1 3 3 0.35 0.7 10 100 200 0 400 3 800 3
python p2.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3 best
#python cnnSample.py relu 2 512 100 300 sgd 1 3 3 0.25 0.5 10 50 100 0 200 3 400 3
#python cnnSample.py relu 2 256 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3
#python cnnSample.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 256 3
# semi
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3 5000 30
#dropout
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.25 0.5 10 50 100 0 200 3 400 3 5000 30
#denseUnit
#python semi.py relu 2 256 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3 5000 30   50
#elu
#python semi.py elu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3 5000 30    51
#adam
#python semi.py relu 2 512 100 300 adam 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3 5000 30  52
#3 max
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 0 3 5000 30  53~
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 0 3 0 3 5000 30 
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 30 52
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 7000 30 56
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 50  54
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 20837 200
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 70 better
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 90
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 110
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 70 0.9
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 70 0.85
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 5000 70 0.95
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 8000 30 0.9
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 8000 50 0.9
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 8000 50 0.95
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 8000 75 0.9
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 8000 100 0.9
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 50 100 0 200 3 400 3 8000 100 0.9
#python semi.py relu 2 512 100 300 sgd 1 3 3 0.35 0.7 10 32 64 0 128 3 0 3 8000 100 0.95
