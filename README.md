# Residual Scheduling: A New Reinforcement Learning Approach to Solving Job Shop Scheduling Problem

## Installation

Setup the virtual environment.
```c
podman run -it --name={YOUR_NAME}   -v $PWD/ResidualScheduling:/ResidualScheduling pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

Install required packages in the environment.
```c
pip install torch-geometric  opencv-python plotly matplotlib gym tensorboard pandas colorhash
```
## Run training
Follow the example to run a FJSP training procedure(RS). And there are some parameters for ablation studying.
```c
python3 train.py --date=train --instance_type=FJSP --data_size=10 --delete_node=true
```

## Reproduced the result in paper
Follow the example to run a FJSP testing 
```c
python3 test.py --date=test --instance_type=FJSP --delete_node=true --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS_FJSP/best'
```
Follow the example to run a FJSP testing (RS+op)
```c
python3 test.py --date=test --instance_type=FJSP --test_dir='./datasets/FJSP/Brandimarte_Data' --load_weight='./weight/RS+op_FJSP/best'
```

### Similarly, for JSP
Follow the example to run a JSP testing (RS)
```c
python3 test.py --date=test --instance_type=JSP --delete_node=true --test_dir='./datasets/JSP/public_benchmark/ta' --load_weight='./weight/RS_JSP/best'
```
Follow the example to run a JSP testing (RS+op)
```c
python3 test.py --date=test --instance_type=JSP --test_dir='./datasets/JSP/public_benchmark/ta' --load_weight='./weight/RS+op_JSP/best'
```

## Hyperparameters list
```c
    python3 train.py \
    --device='cuda' \
    --instance_type='FJSP' \
    --data_size=10 \
    --max_process_time=100 \
    --delete_node=False \
    --entropy_coef=1e-2 \
    --episode=300001 \
    --lr=1e-4 \
    --step_size=1000 \
    --hidden_dim=256 \
    --GNN_num_layers=3 \
    --policy_num_layers=2 \
    --date='Dummy' \
    --detail=None \
    --test_dir='./datasets/FJSP/Brandimarte_Data' \
    --load_weight='./weight/RS_FJSP/best'
```