import bisect
import random
from env.utils.mach_job_op import *
from env.utils.generator import *
from env.utils.graph import Graph
import torch
import time

class JSP_Instance:
    def __init__(self, args):
        self.args, self.process_time_range = args, [1, args.max_process_time]

        self.job_num, self.machine_num = 0, 0

        self.jobs, self.machines = [], []
        self.time_stamp = []
        self.current_time = 0
        self.max_process_time = 0

        self.graph = None

    def generate_case(self):
        self.insert_jobs(job_num=self.job_num)
        
    def insert_jobs(self, job_num):
        self.register_time(0)
        if self.args.instance_type == 'FJSP':
            for i in range(job_num):
                job_id = len(self.jobs)
                op_config = gen_operations_FJSP(self.machine_num, self.process_time_range)
                self.jobs.append(Job(args=self.args, job_id=job_id, op_config=op_config))

        elif self.args.instance_type == 'JSP':
            for i in range(job_num):
                job_id = len(self.jobs)
                op_config = gen_operations_JSP(self.machine_num, self.machine_num, self.process_time_range)
                self.jobs.append(Job(args=self.args, job_id=job_id, op_config=op_config))

        # build up the graph
        self.graph = Graph(self.args, self.job_num, self.machine_num)
        for i in range(job_num):
            self.graph.add_job(self.jobs[i])
        
    def reset(self):
        self.job_num = random.randint(3, self.args.data_size)
        self.machine_num = random.randint(3, self.args.data_size)

        self.jobs = []
        self.machines = [Machine(machine_id) for machine_id in range(self.machine_num)]
        self.current_time = 0
        self.time_stamp = []
        self.generate_case()

    def load_instance(self, filename):
        self.jobs = []
        self.current_time = 0
        self.time_stamp = []
        
        f = open(filename)
        line = f.readline()
        while line[0] == '#':
            line = f.readline()
            
        line = line.split()
        self.job_num, self.machine_num = int(line[0]), int(line[1])
        self.machines = [Machine(machine_id) for machine_id in range(self.machine_num)]

        if self.args.instance_type == "JSP":
            for i in range(self.job_num):
                op_config = []
                line = f.readline().split()
                for j in range(self.machine_num):
                    machine_id, process_time = int(line[j * 2]), int(line[j * 2 + 1])
                    if process_time == 0:
                        continue
                    machine_and_processtime = [(machine_id, process_time)]
                    op_config.append({"id": j, "machine_and_processtime": machine_and_processtime})
                self.jobs.append(Job(args=self.args, job_id=i, op_config=op_config))
        
        if self.args.instance_type == "FJSP":
            for i in range(self.job_num):
                op_config = []
                line = f.readline().split()
                op_num = int(line[0])
                cur = 1
                for j in range(op_num):
                    machine_and_processtime = []
                    machine_num = int(line[cur])
                    cur += 1
                    for _ in range(machine_num):
                        machine_id, process_time = int(line[cur]), int(line[cur + 1])
                        machine_and_processtime.append((machine_id - 1, process_time))
                        cur += 2
                    op_config.append({"id": j, "machine_and_processtime": machine_and_processtime})
                self.jobs.append(Job(args=self.args, job_id=i, op_config=op_config))

        self.graph = Graph(self.args, self.job_num, self.machine_num)
        for i in range(self.job_num):
            self.graph.add_job(self.jobs[i])
        
        self.register_time(0)

    def done(self):
        for job in self.jobs:
            if job.done() == False:
                return False
        return True
    
    def get_graph_data(self):
        self.graph.update_feature(self.jobs, self.machines, self.current_time)
        data, op_unfinished = self.graph.get_data()
        return data.to(self.args.device), op_unfinished
        
    def assign(self, step_op):
        job_id, op_id = step_op['job_id'], step_op['op_id']
        op_info = {
            "job_id"        : job_id,
            "op_id"         : op_id,
            "current_time"  : max(self.current_time, self.jobs[job_id].current_op().avai_time),
            "process_time"  : step_op['process_time']
        }
        op_finished_time = self.machines[step_op['m_id']].process_op(op_info)
        self.jobs[job_id].current_op().update(self.current_time, step_op['process_time'])
        if self.jobs[job_id].next_op() != -1:
            self.jobs[job_id].update_current_op(avai_time=op_finished_time)
        self.register_time(op_finished_time)

        if self.args.delete_node:
            self.graph.remove_node(job_id, self.jobs[job_id].operations[op_id])

    def register_time(self, time):
        index = bisect.bisect_left(self.time_stamp, time)
        if index == len(self.time_stamp) or self.time_stamp[index] != time:
            self.time_stamp.insert(index, time)
    
    def update_time(self):
        self.current_time = self.time_stamp.pop(0)
    
    def current_avai_ops(self):
        if self.done() == True:
            return None

        avai_ops = []
        avai_mat = np.zeros((self.machine_num, self.job_num),)

        for m in self.machines:
            if m.avai_time() > self.current_time:
                continue
            for job in self.jobs:
                if job.done() == True or job.current_op().avai_time > self.current_time:
                    continue

                for machine_and_processtime in job.current_op().machine_and_processtime:
                    if m.machine_id == machine_and_processtime[0]:
                        avai_mat[machine_and_processtime[0]][job.job_id] = machine_and_processtime[1]

        for i in range(self.job_num):
            avai_m_idx = np.nonzero(avai_mat[:,i])
            if len(avai_m_idx) == 1 and np.count_nonzero(avai_mat[avai_m_idx[0]]) == 1: # 1<->1 op<->m, it must be assigned
                self.assign({
                    'm_id'          : avai_m_idx[0].item(),
                    'job_id'        : i,
                    'op_id'         : self.jobs[i].current_op_id,
                    'node_id'       : self.jobs[i].current_op().node_id,
                    'process_time'  : avai_mat[avai_m_idx[0].item()][i]
                })
                avai_mat[avai_m_idx[0].item()][i] = 0
        
        candidates = np.where(avai_mat > 0)
        for i in range(len(candidates[0])):
            avai_ops.append({
                'm_id'          : candidates[0][i],
                'job_id'        : candidates[1][i],
                'op_id'         : self.jobs[candidates[1][i]].current_op_id,
                'node_id'       : self.jobs[candidates[1][i]].current_op().node_id,
                'process_time'  : avai_mat[candidates[0][i]][candidates[1][i]]
            })
        if len(avai_ops) == 0:
            self.update_time()
            return self.current_avai_ops()
        else:
            return avai_ops
        
    def get_max_process_time(self):
        return self.graph.get_max_process_time()
