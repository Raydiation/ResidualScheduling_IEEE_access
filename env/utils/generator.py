import numpy as np
import random
import os
import time

#JSP
def gen_operations_JSP(op_num, machine_num, op_process_time_range): 
    op = []
    m_seq = [i for i in range(machine_num)]
    random.shuffle(m_seq)
    for op_id in range(op_num):
        process_time = np.random.randint(*op_process_time_range)
        mach_ptime = [(m_seq[op_id], process_time)]
        op.append({"id": op_id, "machine_and_processtime": mach_ptime})
    return op

# FJSP
def gen_operations_FJSP(machine_num, op_process_time_range): 
    op = []
    
    op_num = random.randint(int(0.8 * machine_num), int(1.2 * machine_num))

    for op_id in range(op_num):
        random_size = np.random.choice(range(1, machine_num + 1, 1)) # the number of usable machine for this operation
        m_id = sorted(np.random.choice(machine_num, size=random_size, replace=False)) # the set of index of usable machine id with size random_size
        mach_ptime = []
        for id in m_id:
            process_time = np.random.randint(*op_process_time_range)
            mach_ptime.append((id, process_time))
        op.append({"id": op_id, "machine_and_processtime": mach_ptime})
    return op


