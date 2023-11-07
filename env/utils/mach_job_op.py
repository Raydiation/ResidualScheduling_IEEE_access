import numpy as np
MAX = 1e6

AVAILABLE = 0
PROCESSED = 1
COMPLETED = 3
FUTURE = 2

class Machine:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.processed_op_history = []
    
    def process_op(self, op_info):
        machine_avai_time = self.avai_time()
        start_time = max(op_info["current_time"], machine_avai_time)
        assert start_time == op_info["current_time"]
        op_info["start_time"] = start_time
        finished_time = start_time + op_info["process_time"]
        self.processed_op_history.append(op_info)
        return finished_time
        
    def avai_time(self):
        if len(self.processed_op_history) == 0:
            return 0
        else:
            return self.processed_op_history[-1]["start_time"] + self.processed_op_history[-1]["process_time"]

    def get_status(self, current_time):
        if current_time >= self.avai_time():
            return AVAILABLE
        else:
            return PROCESSED

class Job:
    def __init__(self, args, job_id, op_config):
        self.args = args
        self.job_id = job_id
        self.operations = [Operation(self.args, self.job_id, config) for config in op_config]
        self.op_num = len(op_config)

        self.current_op_id = 0 

        self.acc_expected_process_time = [0]
        for op in self.operations[::-1]:
            self.acc_expected_process_time.append(self.acc_expected_process_time[-1] + op.expected_process_time)
        self.acc_expected_process_time = self.acc_expected_process_time[::-1]
        
    def current_op(self):
        if self.current_op_id == -1:
            return None
        else:
            return self.operations[self.current_op_id]
    
    def update_current_op(self, avai_time):
        self.operations[self.current_op_id].avai_time = avai_time 
    
    def next_op(self):
        if self.current_op_id + 1 < self.op_num:
            self.current_op_id += 1
        else:
            self.current_op_id = -1
        return self.current_op_id
    
    def done(self):
        if self.current_op_id == -1:
            return True
        else:
            return False

class Operation:
    def __init__(self, args, job_id, config):
        self.args = args
        self.job_id = job_id
        self.op_id = config['id']
        self.machine_and_processtime = config['machine_and_processtime']
        self.node_id = -1
        if self.op_id == 0:
            self.avai_time = 0
        else:
            self.avai_time = MAX

        self.start_time = -1 
        self.finish_time = -1 

        total = 0
        for pair in self.machine_and_processtime:
            total += pair[1]
        self.expected_process_time = total / len(self.machine_and_processtime)
        
    def update(self, start_time, process_time):
        self.start_time = start_time
        self.finish_time = start_time + process_time
    
    def get_status(self, current_time):
        if self.start_time == -1:
            if current_time >= self.avai_time:
                return AVAILABLE
            else:
                return FUTURE
        else:
            if current_time >= self.finish_time:
                return COMPLETED
            else:
                return PROCESSED