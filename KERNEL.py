import pyopencl as cl
import numpy as np

class table_class:
    def __init__(self, context, table_input = None, table_shape = None, inp = True, outp = True):
        self.np_array = table_input
        self.cl_buffer = None
        if (not inp) and (not outp):
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)
        elif inp and (not outp):
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)
        elif (not inp) and outp:
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)
        else:
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)

    def cpu_to_gpu(self, queue):
        cl.enqueue_copy(queue, self.cl_buffer, self.np_array).wait()
    
    def gpu_to_cpu(self, queue):
        cl.enqueue_copy(queue, self.np_array, self.cl_buffer).wait()

def find_globals(kernel_str):
    kernel_str = kernel_str[1:]
    kernel_args = kernel_str.split(')')
    args_list = kernel_args[0].split(',')
        
    for i in range(len(args_list)):
        if '*' in args_list[i]:
            args_list[i] = '__global ' + args_list[i].strip()

    kernel_args[0] = ','.join(args_list)
    return  ')'.join(kernel_args)

class program_class:
    def __init__(self, context, kernel_str, global_size, local_size = None, simple_kernel = True):
        if simple_kernel:
            kernel_str = "__kernel void func(" + find_globals(kernel_str)
            print("This is a simple kernel with only one func()!")
        self.program = cl.Program(context, kernel_str).build()
        self.global_size = global_size
        self.local_size = local_size

    def run_program(self, queue, *args):
        args_cleared = []
        for arg in args:
            if isinstance(arg, table_class):
                args_cleared.append(arg.cl_buffer)
            else:
                args_cleared.append(arg)
        self.program.func(queue, self.global_size, self.local_size, *args_cleared).wait()

class programs_ios_class:
    def __init__(self):
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.programs = {}
        self.ios = {}

    def new_table(self, name, table_input = None, shape = None, dtype = None):
        if table_input is None:
            self.ios[name] = table_class(self.context, table_shape = shape, table_dtype = dtype)
        else:
            self.ios[name] = table_class(self.context, table_input = table_input)

    def table(self, name):
        return self.ios[name].np_array.view()
    
    def del_table(self, name):
        if name in self.ios:
            del self.ios[name]
        else:
            print(f"Table '{name}' not found in the dictionary!")

    def del_all_tables(self):
        self.ios = {}

    def new_program(self, name, kernel_str, global_size, local_size = None):
        self.programs[name] = program_class(self.context, kernel_str, global_size, local_size)

    def run_program(self, name, *args, inp = True, outp = True):
        selected_program = self.programs[name]
        ios_cleared = []
        for arg in args:
            if arg in self.ios:
                ios_cleared.append(self.ios[arg])
            else:
                ios_cleared.append(arg)
        if inp:
            for io in ios_cleared:
                if isinstance(io, table_class):
                    io.cpu_to_gpu(self.queue)
        selected_program.run_program(self.queue, *ios_cleared)
        if outp:
            for io in ios_cleared:
                if isinstance(io, table_class):
                    io.gpu_to_cpu(self.queue)
