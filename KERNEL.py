import pyopencl as cl
import numpy as np
import re

class table_class:
    def __init__(self, context, table_shape, table_dtype, inp = True, outp = True):
        self.np_array = np.zeros(shape=(table_shape), dtype = table_dtype)
        self.cl_buffer = None
        if inp and (not outp):
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)
        if outp and (not inp):
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)
        if inp and outp:
            self.cl_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.np_array)

    def cpu_to_gpu(self, queue):
        cl.enqueue_copy(queue, self.cl_buffer, self.np_array).wait()
    
    def gpu_to_cpu(self, queue):
        cl.enqueue_copy(queue, self.np_array, self.cl_buffer).wait()

class program_class:
    def __init__(self, context, kernel_str, global_size, local_size = None):
        kernel_str = "__kernel void func" + re.sub(r'\(([^)]*\*\s*\w+[^)]*)\)', r'(__global \1)', kernel_str) # generaly a kernels can have multiple functions like the void func but for simpl. it cannot!
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

    def new_table(self, name, shape, dtype):
        self.ios[name] = table_class(self.context, shape, dtype)

    def table(self, name):
        return self.ios[name].np_array.view()

    def del_table(self, name):
        self.ios[name].remove()
        del self.ios[name]

    def new_program(self, name, kernel_str, global_size, local_size = None):
        self.programs[name] = program_class(self.context, kernel_str, global_size, local_size)

    def del_program(self, name):
        self.programs[name].remove()
        del self.programs[name]

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
