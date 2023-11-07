import torch
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
import time
import os

def test():
    for instance in os.listdir(args.test_dir):
        file = os.path.join(args.test_dir, instance)
        avai_ops = env.load_instance(file)
        st = time.time()

        while True:
            data, op_unfinished= env.get_graph_data()
            action_idx, action_prob = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time, greedy=True)
            avai_ops, _, done = env.step(avai_ops[action_idx])
            
            if done:
                ed = time.time()
                policy.clear_memory()

                print("instance : {}, ms : {}, time : {}".format(file, env.get_makespan(), ed - st))
                with open("./result/{}/test_result.txt".format(args.date),"a") as outfile:
                    outfile.write(f'instance : {file:60}, policy : {env.get_makespan():10}\t')
                    outfile.write(f'time : {ed - st:10}\n')
                break

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)
    
    policy.load_state_dict(torch.load(args.load_weight, map_location=args.device), False)
    with torch.no_grad():
        test()
                    