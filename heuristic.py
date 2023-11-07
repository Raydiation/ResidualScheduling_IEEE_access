import numpy as np
import random

MAX = 1e6

def heuristic_makespan(env, avai_ops, rule):
    if rule == "MOR":
        while True:
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)

            avai_ops, _, done = env.step(avai_ops[action_idx])

            if done:
                return env.get_makespan()
    if rule == "FIFO":
         while True:
            action_idx = FIFO(avai_ops, env.jsp_instance.jobs)

            avai_ops, _, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()
    if rule == "SPT":
         while True:
            action_idx = SPT(avai_ops)

            avai_ops, _, done = env.step(avai_ops[action_idx])
            if done:
                return env.get_makespan()

    if rule == "MWKR":
        while True:
            action_idx = MWKR(avai_ops, env.jsp_instance.jobs)

            avai_ops, _, done = env.step(avai_ops[action_idx])

            if done:
                return env.get_makespan()

def rollout(env, avai_ops):
    epsilon = 0.1
    while True:
        magic_num = random.random()
        if magic_num < epsilon:
            action_idx = Random(avai_ops)
        else:
            action_idx = MOR(avai_ops, env.jsp_instance.jobs)
        avai_ops, done = env.step(avai_ops, action_idx)
        if done:
            return env.get_makespan()


def Random(avai_ops):
    return np.random.choice(len(avai_ops), size=1)[0]

def MOR(avai_ops, jobs):
    max_remaining_op = -1
    action_idx = -1

    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        job = jobs[op_info['job_id']]

        if len(job.operations) - op_info['op_id'] >= max_remaining_op:
            action_idx = i
            max_remaining_op = len(job.operations) - op_info['op_id']
            
    return action_idx

def MWKR(avai_ops, jobs):
    action_idx = -1
    max_work_remaining = -1
    
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]
        job = jobs[op_info['job_id']]
        if job.acc_expected_process_time[op_info['op_id']] > max_work_remaining:
            max_work_remaining = job.acc_expected_process_time[op_info['op_id']]
            action_idx = i
            
    return action_idx


def FIFO(avai_ops, jobs):
    min_avai_time = MAX
    action_idx = -1

    for i in range(len(avai_ops)): 

        op_info = avai_ops[i]
        op = jobs[op_info['job_id']].operations[op_info['op_id']]

        if op.avai_time < min_avai_time:
            action_idx = i
            min_avai_time = op.avai_time

    return action_idx


def SPT(avai_ops):
    min_process_time = MAX
    action_idx = -1
    
    for i in range(len(avai_ops)):
        op_info = avai_ops[i]

        if op_info['process_time'] < min_process_time:
            action_idx = i
            min_process_time = op_info['process_time']

    return action_idx