import bisect
import random
from env.utils.mach_job_op import *
from env.utils.generator import *
from env.utils.graph import Graph
import torch
import time

class JSP_Instance:
    def __init__(self, args):
        self.args = args
        self.process_time_range = [1, args.max_process_time]

        self.job_num = 0
        self.machine_num = 0
        self.op_num = 0

        self.jobs = []
        self.machines = []
        self.arrival_time = 0
        self.current_time = 0
        self.time_stamp = []
        self.graph = None
        self.max_process_time = 0
        
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

        self.graph = Graph(self.args, self.job_num, self.machine_num)
        for i in range(job_num):
            self.graph.add_job(self.jobs[i])

        self.graph.convert_to_tensor()

        self.graph.op_unfinished = [i for i in range(self.graph.op_num)]
        
    def reset(self):
        self.job_num = random.randint(3, self.args.data_size)
        self.machine_num = random.randint(3, self.args.data_size)

        self.jobs = []
        self.machines = [Machine(machine_id) for machine_id in range(self.machine_num)]
        self.arrival_time = 0
        self.current_time = 0
        self.time_stamp = []
        self.generate_case()

    def load_instance(self, filename):
        self.jobs = []
        self.arrival_time = 0
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
                    mach_ptime = [(machine_id, process_time)]
                    op_config.append({"id": j, "machine_and_processtime": mach_ptime})
                self.jobs.append(Job(args=self.args, job_id=i, op_config=op_config))
        
        if self.args.instance_type == "FJSP":
            for i in range(self.job_num):
                op_config = []
                line = f.readline().split()
                op_num = int(line[0])
                cur = 1
                for j in range(op_num):
                    mach_ptime = []
                    machine_num = int(line[cur])
                    cur += 1
                    for _ in range(machine_num):
                        machine_id, process_time = int(line[cur]), int(line[cur + 1])
                        mach_ptime.append((machine_id - 1, process_time))
                        cur += 2
                    op_config.append({"id": j, "machine_and_processtime": mach_ptime})
                self.jobs.append(Job(args=self.args, job_id=i, op_config=op_config))

        self.graph = Graph(self.args, self.job_num, self.machine_num)
        for i in range(self.job_num):
            self.graph.add_job(self.jobs[i])

        self.graph.op_unfinished = [i for i in range(self.graph.op_num)]
        self.graph.convert_to_tensor()
        
        self.register_time(0)

    def done(self):
        for job in self.jobs:
            if job.done() == False:
                return False
        return True
    
    def current_avai_ops(self):
        avai_ops = self.available_ops()
        return avai_ops
    
    def get_graph_data(self):
        self.graph.update_feature(self.jobs, self.machines, self.current_time)
        data, op_unfinished = self.graph.get_data()
        data = data.to(self.args.device)
        return data, op_unfinished
        
    def assign(self, step_op):
        job_id, op_id = step_op['job_id'], step_op['op_id']
        assert op_id == self.jobs[job_id].current_op_id
        op = self.jobs[job_id].current_op()
        op_info = {
            "job_id": job_id,
            "op_id": op_id,
            "current_time": max(self.current_time, op.avai_time),
            "process_time": step_op['process_time']
        }
        op_finished_time = self.machines[step_op['m_id']].process_op(op_info)
        self.jobs[job_id].current_op().update(self.current_time, step_op['process_time'])
        if self.jobs[job_id].next_op() != -1:
            self.jobs[job_id].update_current_op(avai_time=op_finished_time)
        self.register_time(op_finished_time)

        op.selected_machine_id = int(step_op['m_id'])
        op.process_time = step_op['process_time']

    def register_time(self, time):
        bisect.insort(self.time_stamp, time)
    
    def update_time(self):
        self.current_time = self.time_stamp.pop(0)
    
    def available_ops(self):
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

                for mach_ptime in job.current_op().machine_and_processtime:
                    if m.machine_id == mach_ptime[0]:
                        avai_mat[mach_ptime[0]][job.job_id] = mach_ptime[1]

        for i in range(self.job_num):
            avai_m_idx = np.nonzero(avai_mat[:,i])
            if len(avai_m_idx) == 1 and np.count_nonzero(avai_mat[avai_m_idx[0]]) == 1:
                self.assign({
                    'm_id' : avai_m_idx[0].item(),
                    'process_time' : avai_mat[avai_m_idx[0].item()][i],
                    'job_id' : i,
                    'op_id' : self.jobs[i].current_op_id,
                    'node_id' : self.jobs[i].current_op().node_id
                })
                avai_mat[avai_m_idx[0].item()][i] = 0
        
        candidates = np.where(avai_mat > 0)
        for i in range(len(candidates[0])):
            avai_ops.append({
                'm_id' : candidates[0][i],
                'process_time' : avai_mat[candidates[0][i]][candidates[1][i]],
                'job_id' : candidates[1][i],
                'op_id' : self.jobs[candidates[1][i]].current_op_id,
                'node_id' : self.jobs[candidates[1][i]].current_op().node_id
            })
        if len(avai_ops) == 0:
            self.update_time()
            return self.available_ops()
        else:
            return avai_ops
        
    def get_max_process_time(self):
        return self.graph.get_max_process_time()
