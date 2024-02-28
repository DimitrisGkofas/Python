from KERNEL import programs_ios_class
from kernels_code import func
import numpy as np
'''
gpu_programs = programs_ios_class()
table_pos = gpu_programs.new_table('pos', (8,), np.float32)
gpu_programs.new_program('progr_1', func, (8,))

gpu_programs.table('pos')[0] = np.float32(8)
gpu_programs.table('pos')[1] = np.float32(1)
gpu_programs.run_program('progr_1', 'pos', np.float32(1.5))
print(gpu_programs.table('pos'))
'''
