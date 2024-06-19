import torch
from torch_geometric.data import HeteroData
import numpy as np
import time
from bisect import bisect_left

AVAILABLE = 0
PROCESSED = 1
COMPLETE = 3
FUTURE = 2

def binary_search(list_, target):
    left, right = 0, len(list_)
    pos = bisect_left(list_, target, left, right)
    return pos if pos != right and list_[pos] == target else -1

class Graph:
    def __init__(self, args, job_num, machine_num):

        self.op_op_edge_src_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        self.op_op_edge_tar_idx = torch.empty(size=(1,0), dtype=torch.int64)                    # for op<->op
        self.op_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                           # for op<->m
        self.m_edge_idx = torch.empty(size=(1,0), dtype=torch.int64)                            # for op<->m
        self.m_m_edge_idx = torch.tensor([[i for i in range(machine_num)]], dtype=torch.int64)  # for m<->m
        self.edge_x = torch.empty(size=(1,0), dtype=torch.int64)

        self.op_x = []
        self.m_x = []

        self.args = args
        self.job_num = job_num
        self.machine_num = machine_num
        self.op_num = 0
        self.op_unfinished = []
        self.current_op = [] 

        self.max_process_time = 0.

    def get_data(self):
        data = HeteroData()

        data['op'].x    = torch.FloatTensor(self.op_x)
        data['m'].x     = torch.FloatTensor(self.m_x)

        data['op', 'to', 'op'].edge_index   = torch.cat((self.op_op_edge_src_idx, self.op_op_edge_tar_idx), dim=0).contiguous()
        data['op', 'to', 'm'].edge_index    = torch.cat((self.op_edge_idx, self.m_edge_idx), dim=0).contiguous()
        data['m', 'to', 'op'].edge_index    = torch.cat((self.m_edge_idx, self.op_edge_idx), dim=0).contiguous()
        data['m', 'to', 'm'].edge_index     = torch.cat((self.m_m_edge_idx, self.m_m_edge_idx), dim=0).contiguous()

        return data, self.op_unfinished
       
    def add_job(self, job):
        src, tar = self.fully_connect(self.op_num, job.op_num)
        self.op_op_edge_src_idx = torch.cat((self.op_op_edge_src_idx, src.unsqueeze(0)), dim=1)
        self.op_op_edge_tar_idx = torch.cat((self.op_op_edge_tar_idx, tar.unsqueeze(0)), dim=1)
        self.current_op.append(0)
        
        for i in range(job.op_num):
            job.operations[i].node_id = self.op_num # set index of an op in the graph
            op = job.operations[i]
            self.op_edge_idx    = torch.cat((self.op_edge_idx,  torch.tensor([[self.op_num for _ in range(len(op.machine_and_processtime))]])), dim=1)
            self.m_edge_idx     = torch.cat((self.m_edge_idx,   torch.tensor([[machine_and_processtime[0] for machine_and_processtime in op.machine_and_processtime]])), dim=1)
            self.edge_x         = torch.cat((self.edge_x,       torch.tensor([[machine_and_processtime[1] for machine_and_processtime in op.machine_and_processtime]])), dim=1)

            self.op_unfinished.append(self.op_num)
            self.op_num += 1

    def update_feature(self, jobs, machines, current_time):
        self.op_x, self.m_x = [], []
        self.max_process_time = self.get_max_process_time()
        # op feature
        # [status(2/4), exp process time, waiting time, remaining job]
        for i in range(len(jobs)):
            for j in range(self.current_op[i], len(jobs[i].operations)):
                op = jobs[i].operations[j]
                status = op.get_status(current_time)
                if self.args.delete_node == True:
                    feat = [0] * 2
                    feat[status // 2] = 1
                else:
                    feat = [0] * 4
                    feat[status] = 1

                feat.append(op.expected_process_time / self.max_process_time)

                if status == AVAILABLE:
                    feat.append((current_time - op.avai_time) / self.max_process_time)
                else:
                    feat.append(0)

                feat.append(jobs[i].acc_expected_process_time[op.op_id] / jobs[i].acc_expected_process_time[0])

                self.op_x.append(feat) 
        # machine feature
        # [status(2), time to comlete, waiting time]
        for m in machines:
            feat = [0] * 2
            status = m.get_status(current_time)
            feat[status] = 1
            
            if status == AVAILABLE:
                feat.append(0)
                feat.append((current_time - m.avai_time()) / self.max_process_time)
            else:
                feat.append((m.avai_time() - current_time) / self.max_process_time)
                feat.append(0)

            self.m_x.append(feat)
        
    def remove_node(self, job_id, remove_op):
        idx = binary_search(self.op_unfinished, remove_op.node_id)
        self.op_unfinished.pop(idx)
        self.current_op[job_id] += 1

        # op-op
        src_remove_idxs = torch.where(self.op_op_edge_src_idx == idx)[1]
        tar_remove_idxs = torch.where(self.op_op_edge_tar_idx == idx)[1]

        mask = torch.ones(self.op_op_edge_src_idx.shape[1], dtype=bool)
        mask[src_remove_idxs] = False
        mask[tar_remove_idxs] = False

        self.op_op_edge_src_idx = self.op_op_edge_src_idx[:, mask]
        self.op_op_edge_tar_idx = self.op_op_edge_tar_idx[:, mask]

        #op-m, m-op
        remove_idxs = torch.where(self.op_edge_idx == idx)[1]

        mask = torch.ones(self.op_edge_idx.shape[1], dtype=bool)
        mask[remove_idxs] = False

        self.op_edge_idx    = self.op_edge_idx[:, mask]
        self.m_edge_idx     = self.m_edge_idx[:, mask]
        self.edge_x         = self.edge_x[:, mask]

        _, self.op_edge_idx         = torch.unique(self.op_edge_idx, return_inverse=True)
        _, self.op_op_edge_src_idx  = torch.unique(self.op_op_edge_src_idx, return_inverse=True)
        _, self.op_op_edge_tar_idx  = torch.unique(self.op_op_edge_tar_idx, return_inverse=True)
        
        self.op_num -= 1

    def fully_connect(self, begin, size):
        adj_matrix = torch.ones((size, size))
        idxs = torch.where(adj_matrix > 0)
        return idxs[0] + begin, idxs[1] + begin

    def get_max_process_time(self):
        return torch.max(self.edge_x).item()