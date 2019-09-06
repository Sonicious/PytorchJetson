#!/bin/bash
for lr in 0.2 0.1 0.05 0.02 0.01
do
    for mom in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python ownCifar100Trainer.py --no-logging --lr ${lr} --momentum ${mom} --epoch 15
    done
done