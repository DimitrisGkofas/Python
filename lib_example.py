from KERNEL import programs_ios_class
from kernels_code import func
import numpy as np

gpu_programs = programs_ios_class() # make an instance of the main class
table_pos = gpu_programs.new_table('pos', table_input = np.zeros(shape = (8), dtype = np.float32) # make an input 1dimentional table with 8cells
gpu_programs.new_program('progr_1', func, (8,)) # make the program based on the input string function with 8 compute units on one axes the axes X or arange 8 cus in 1d

gpu_programs.table('pos')[0] = np.float32(8) # set some values as like you do in numpy
gpu_programs.table('pos')[1] = np.float32(1) # set some values as like you do in numpy
gpu_programs.run_program('progr_1', 'pos', np.float32(1.5)) # run the chosen program based on its name with the input var names
print(gpu_programs.table('pos')) # last print the outcome!
