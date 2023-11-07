import numpy as np
import copy
import os
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from params import get_args
from env.env import JSP_Env
from model.REINFORCE import REINFORCE
from heuristic import *
from torch.utils.tensorboard import SummaryWriter
import json
import time

MAX = float(1e6)

def train():
    print("start Training")
    best_valid_makespan = MAX

    for episode in range(0, args.episode):
        if episode % 1000 == 0:
            torch.save(policy.state_dict(), "./weight/{}/{}".format(args.date, episode))

        action_probs = []
        avai_ops = env.reset()
        while avai_ops is None:
            avai_ops = env.reset()

        MWKR_ms = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)

        while True:
            MWKR_baseline = heuristic_makespan(copy.deepcopy(env), copy.deepcopy(avai_ops), args.rule)
            baseline = MWKR_baseline - env.get_makespan()

            data, op_unfinished = env.get_graph_data()
            action_idx, action_prob = policy(avai_ops, data, op_unfinished, env.jsp_instance.graph.max_process_time)
            avai_ops, reward, done = env.step(avai_ops[action_idx])

            policy.rewards.append(-reward)
            policy.baselines.append(baseline)
            action_probs.append(action_prob)
            
            if done:
                optimizer.zero_grad()
                loss, policy_loss, entropy_loss = policy.calculate_loss(args.device)
                loss.backward()

                if episode % 10 == 0:
                    writer.add_scalar("action prob", np.mean(action_probs),episode)
                    writer.add_scalar("loss", loss, episode)
                    writer.add_scalar("policy_loss", policy_loss, episode)
                    writer.add_scalar("entropy_loss", entropy_loss, episode)
                
                optimizer.step()
                scheduler.step()

                policy.clear_memory()
                ms = env.get_makespan()
                improve = MWKR_ms - ms
                print("Date : {} \t\t Episode : {} \t\tJob : {} \t\tMachine : {} \t\tPolicy : {} \t\tImprove: {} \t\t MWKR : {}".format(
                    args.date, episode, env.jsp_instance.job_num, env.jsp_instance.machine_num, 
                    ms, improve, MWKR_ms))
                break

if __name__ == '__main__':
    args = get_args()
    print(args)

    os.makedirs('./result/{}/'.format(args.date), exist_ok=True)
    os.makedirs('./weight/{}/'.format(args.date), exist_ok=True)

    with open("./result/{}/args.json".format(args.date),"a") as outfile:
        json.dump(vars(args), outfile, indent=8)

    env = JSP_Env(args)
    policy = REINFORCE(args).to(args.device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.99)
    writer = SummaryWriter(comment=args.date)

    train()
    