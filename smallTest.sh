#!/bin/bash

python ownCifarResNet.py -e 1 --save-model --log-interval 100
python ownCifarAlexNet.py -e 1 --save-model --log-interval 100
python ownCifarDenseNet.py -e 1 --save-model --log-interval 100