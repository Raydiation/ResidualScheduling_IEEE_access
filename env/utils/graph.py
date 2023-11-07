from token import LBRACE
import numpy as np
import torch
from torch_geometric.data import HeteroData
from heuristic import MAX
import time
import bisect

AVAILABLE = 0
PROCESSED = 1
COMPLETE = 3
FUTURE = 2

def bsearch(a, left, right, x) :
    while left <= right :
        mid = (left + right) // 2        
        if a[mid] == x:
            return mid 
        if a[mid] < x:
            left = mid + 1 
        else:
            right = mid - 1         
    return -1

class Graph:
    def __init__(self, args, job_num, machine_num):

        self.op_op_edge_src_idx = np.empty(shape=(0,1))
        self.op_op_edge_tar_idx = np.empty(shape=(0,1))
        self.op_edge_idx = np.empty(shape=(0,1))
        self.m_edge_idx = np.empty(shape=(0,1))
        self.m_m_edge_idx = []

        self.op_x = []
        self.m_x = []

        self.edge_x = []

        for i in range(machine_num):
            self.m_m_edge_idx.append([i, i])

        self.args = args
        self.job_num = job_num
        self.machine_num = machine_num
        self.op_num = 0
        self.op_unfinished = {}
        self.current_op = [0] * job_num 

        self.max_process_time = 0.

    def get_data(self):
        data = HeteroData()
        data['op'].x = torch.FloatTensor(self.op_x)
        data['m'].x = torch.FloatTensor(self.m_x)

        data['op', 'to', 'op'].edge_index = torch.stack((self.op_op_edge_src_idx, self.op_op_edge_tar_idx), dim=1).t().contiguous()
        data['op', 'to', 'm'].edge_index = torch.stack((self.op_edge_idx, self.m_edge_idx), dim=1).t().contiguous()
        data['m', 'to', 'op'].edge_index = torch.stack((self.m_edge_idx, self.op_edge_idx), dim=1).t().contiguous()
        data['m', 'to', 'm'].edge_index = torch.LongTensor(self.m_m_edge_idx).t().contiguous()

        return data, self.op_unfinished
       
    def add_job(self, job):
        src, tar = self.fully_connect(self.op_num, job.op_num)
        self.op_op_edge_src_idx = np.append(self.op_op_edge_src_idx, src)
        self.op_op_edge_tar_idx = np.append(self.op_op_edge_tar_idx, tar)
        for i in range(job.op_num):
            job.operations[i].node_id = self.op_num # set index of an op in the graph
            op = job.operations[i]
            for mach_and_ptime in op.machine_and_processtime:
                self.op_edge_idx = np.append(self.op_edge_idx, [self.op_num])
                self.m_edge_idx = np.append(self.m_edge_idx, [mach_and_ptime[0]])
                self.edge_x.append([mach_and_ptime[1]])
            self.op_num += 1

    def update_feature(self, jobs, machines, current_time):
        self.op_x, self.m_x = [], []
        if self.args.delete_node == True:
            for i in range(len(jobs)):
                cur = self.current_op[i]
                for j in range(cur, len(jobs[i].operations)):
                    op = jobs[i].operations[j]

                    status = op.get_status(current_time)

                    if status == PROCESSED or status == COMPLETE:
                        idx = bsearch(self.op_unfinished, 0, len(self.op_unfinished) - 1, op.node_id)
                        
                        if idx == -1:
                            raise "abnormal idx"

                        self.update_graph(idx)
                        self.op_unfinished.remove(op.node_id)
                        self.current_op[i] += 1
                    else:
                        break
        self.convert_to_tensor()
        self.max_process_time = self.get_max_process_time()
        
        # op feature
        for i in range(len(jobs)):
            job = jobs[i]
            for j in range(self.current_op[i], len(jobs[i].operations)):
                op = job.operations[j]
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

                feat.append(job.acc_expected_process_time[op.op_id] / job.acc_expected_process_time[0])

                self.op_x.append(feat) 

        # machine feature
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

    def convert_to_tensor(self):
        self.op_op_edge_src_idx = torch.LongTensor(self.op_op_edge_src_idx)
        self.op_op_edge_tar_idx = torch.LongTensor(self.op_op_edge_tar_idx)
        self.op_edge_idx = torch.LongTensor(self.op_edge_idx)
        self.m_edge_idx = torch.LongTensor(self.m_edge_idx)
        self.m_m_edge_idx = torch.LongTensor(self.m_m_edge_idx)
        self.edge_x = torch.FloatTensor(self.edge_x)
        
    def update_graph(self, idx):
        src_idxs = np.where(self.op_op_edge_src_idx == idx)
        self.op_op_edge_src_idx = np.delete(self.op_op_edge_src_idx, src_idxs)
        self.op_op_edge_tar_idx = np.delete(self.op_op_edge_tar_idx, src_idxs)
        tar_idxs = np.where(self.op_op_edge_tar_idx == idx)
        self.op_op_edge_src_idx = np.delete(self.op_op_edge_src_idx, tar_idxs)
        self.op_op_edge_tar_idx = np.delete(self.op_op_edge_tar_idx, tar_idxs)

        #op-m, m-op
        idxs = np.where(self.op_edge_idx == idx)
        self.op_edge_idx = np.delete(self.op_edge_idx, idxs)
        self.m_edge_idx = np.delete(self.m_edge_idx, idxs)
        self.edge_x = np.delete(self.edge_x, idxs)

        _, self.op_edge_idx = np.unique(self.op_edge_idx, return_inverse=True)
        _, self.op_op_edge_src_idx = np.unique(self.op_op_edge_src_idx, return_inverse=True)
        _, self.op_op_edge_tar_idx = np.unique(self.op_op_edge_tar_idx, return_inverse=True)


    def fully_connect(self, begin, size):
        adj_matrix = np.ones((size, size),)
        idxs = np.where(adj_matrix > 0)
        edge_index = np.stack((idxs[0] + begin, idxs[1] + begin))
        return edge_index[0], edge_index[1]

    def get_max_process_time(self):
        return np.max(self.edge_x.numpy())