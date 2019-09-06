#!/bin/bash

python ownCifarResNet.py -e 20 --save-model --log-interval 100
python ownCifarAlexNet.py -e 20 --save-model --log-interval 100
python ownCifarDenseNet.py -e 20 --save-model --log-interval 100