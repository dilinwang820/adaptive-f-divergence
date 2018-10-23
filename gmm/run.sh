#! /bin/bash

set -e 

if [ "$#" -ne 1 ];then
    echo "Usage run.sh dim"
    exit 1
fi

dim=$1
lr=5e-2
sample_size=256

for((seed=100;seed<120;seed++))
do
    # adapted
    for alpha in -1.0 -0.5 0.5 1.0
    do
        python trainer.py --dim ${dim} --proposal mixture --learning_rate ${lr} --save --clean --method adapted --alpha  ${alpha} --seed ${seed} --sample_size ${sample_size} 
    done
    for alpha in -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0
    do
        python trainer.py --dim ${dim} --proposal mixture --learning_rate ${lr} --save --clean --method alpha --alpha ${alpha} --seed ${seed} --sample_size ${sample_size}
    done
done


