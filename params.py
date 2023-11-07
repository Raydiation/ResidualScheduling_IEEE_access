import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Arguments for RL_GNN_JSP')
    # args for normal setting
    parser.add_argument('--device', type=str, default='cuda')
    # args for env
    parser.add_argument('--instance_type', type=str, default='FJSP')
    parser.add_argument('--data_size', type=int, default=10)
    parser.add_argument('--max_process_time', type=int, default=100, help='Maximum Process Time of an Operation')
    parser.add_argument('--delete_node', type=bool, default=False)
    # args for RL
    parser.add_argument('--entropy_coef', type=float, default=1e-2)
    parser.add_argument('--episode', type=int, default=300001)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--step_size', type=float, default=1000)
    # args for policy network
    parser.add_argument('--hidden_dim', type=int, default=256) #256
    # args for GNN
    parser.add_argument('--GNN_num_layers', type=int, default=3)
    # args for policy
    parser.add_argument('--policy_num_layers', type=int, default=2)
    
    # args for nameing
    parser.add_argument('--date', type=str, default='Dummy')
    parser.add_argument('--detail', type=str, default="no")
    # args for structure
    parser.add_argument('--rule', type=str, default='MWKR')

    # args for val/test
    parser.add_argument('--test_dir', type=str, default='./datasets/FJSP/Brandimarte_Data')
    parser.add_argument('--load_weight', type=str, default='./weight/RS_FJSP/best')

    args = parser.parse_args()
    return args
