#! /bin/bash

set -e 

run_block() {
    method=$1
    alpha=$2
    epochs=$3
    dataset=$4
    K=$5
    n_hidden=$6
    n_splits=$7

    for ((trial=1;trial<=${n_splits};trial++))
    {
        python bnn_trainer.py --method ${method} --alpha ${alpha} --n_epoches ${epochs} --dataset ${dataset} --batch_size 32 --sample_size ${K} --n_hidden ${n_hidden} --trial ${trial} --learning_rate 1e-3 --save
    }
} 


super_block() {

    epochs=$1
    dataset=$2
    K=$3
    n_hidden=$4
    n_splits=$5

    #run_block negcdf -1.0 $epochs $dataset $K $n_hidden $n_splits & p1=$!
    #run_block negcdf -2.0 $epochs $dataset $K $n_hidden $n_splits & p2=$!
    #run_block negcdf -0.5 $epochs $dataset $K $n_hidden $n_splits & p3=$!
    #run_block alpha -1.0 $epochs $dataset $K $n_hidden $n_splits & p4=$!

    #echo "waitting ..." $p1 $p2 $p3 $p4
    #wait $p1 $p2 $p3 $p4

    #run_block alpha 0.0 $epochs $dataset $K $n_hidden $n_splits & p5=$!
    #run_block alpha 0.5 $epochs $dataset $K $n_hidden $n_splits & p6=$!
    #run_block alpha 1.0 $epochs $dataset $K $n_hidden $n_splits & p7=$!
    #run_block alpha 2.0 $epochs $dataset $K $n_hidden $n_splits & p8=$!

    #echo "waitting ..." $p5 $p6 $p7 $p8
    #wait $p5 $p6 $p7 $p8
    #run_block alpha 500.0 $epochs $dataset $K $n_hidden $n_splits
    run_block alpha -1.0 $epochs $dataset $K $n_hidden $n_splits
}

### boston ###
epochs=500
dataset="boston"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### concrete ###
epochs=300
dataset="concrete"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### energy ###
epochs=500
dataset="energy"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### kin8nm ###
epochs=50
dataset="kin8nm"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### naval ###
epochs=50
dataset="naval"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### combined ###
epochs=100
dataset="combined"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### wine ###
epochs=100
dataset="wine"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits

### yacht ###
epochs=500
dataset="yacht"
K=100
n_hidden=50
n_splits=20
super_block $epochs $dataset $K $n_hidden $n_splits


### protein ###
epochs=50
dataset="protein"
K=100
n_hidden=100
n_splits=5
super_block $epochs $dataset $K $n_hidden $n_splits

### year ###
epochs=5
dataset="year"
K=100
n_hidden=100
n_splits=5
super_block $epochs $dataset $K $n_hidden $n_splits

