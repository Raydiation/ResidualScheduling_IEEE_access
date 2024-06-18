import torch
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
import time
import os

MAX = float(1e6)

def eval_(episode, valid_sets=None):
    if args.instance_type == 'FJSP':
        valid_dir = './datasets/FJSP/data_dev'
        valid_sets = ['1510']

    else:
        valid_dir = './datasets/JSP/JSP_validation'
        valid_sets = ['20x20_valid']

    for _set in valid_sets:
        total_ms = 0.
        for instance in sorted(os.listdir(os.path.join(valid_dir, _set))):
            file = os.path.join(os.path.join(valid_dir, _set), instance)

            st = time.time()
            avai_ops = env.load_instance(file)

            while True:
                data, op_unfinished= env.get_graph_data()
                action_idx, _ = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time, greedy=True)
                avai_ops, _, done = env.step(avai_ops[action_idx])

                if done:
                    ed = time.time()
                    ms = env.get_makespan()
                    total_ms += ms
                    policy.clear_memory()

                    print('instance : {}, ms : {}, time : {}'.format(file, ms, ed - st))
                    break
        with open('./result/{}/valid_result_{}.txt'.format(args.date, _set),"a") as outfile:
            outfile.write(' set : {}, episode : {}, avg_ms : {}\n'.format(_set, episode, total_ms / len(os.listdir(os.path.join(valid_dir, _set)))))
        

if __name__ == '__main__':
    args = get_args()
    print(args)
    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)

    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)
    for episode in os.listdir('./weight/{}/'.format(args.date)):
        if episode == 'best':
            continue
        print(f'date : {args.date} episode : {episode}')
        policy.load_state_dict(torch.load('./weight/{}/{}'.format(args.date, episode), map_location=args.device), False)
        with torch.no_grad():
            valid_makespan = eval_(episode, args.valid_sets)