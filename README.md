The following is a simple api with name KERNEL on top of pyopencl api to make you come closer to gpu programming!
It is free to change it and use it as you wish!
Bellow the main obj with name: programs_ios_class is discribed, as it is the main component of KERNEL.py file for you to use:
First make an instance of this class to hold all of your tables and programms.
Second choose in your program objects how many compute units you need to have.
Compute units are the objects tha run your program in the gpu. Each compute unit runs a portion of your program.
Down below is a simple program func to run in your gpu. This is the format to run any program in the gpu based on the opencl api architecture!
func = """(float *positions, float coe) {
    uint x_index = get_global_id(0);
    positions[x_index] = positions[x_index] * coe;
}
"""
tables can be distinguished via the *.
Simple variables that are the same for the hole program don't have *.
The last step is for you to chooseyour compute units numbers and dimentions.
If you want 1d data to be accesed in a row, you can use something like that in your code: uint x_index = get_global_id(0);
If you want 2d data to be accesed as a table, you can use something like that in your program: 
uint x_index = get_global_id(0);
uint y_index = get_global_id(1);
uint oneD_index = x_index + y_index * tables_x_dim;
As you already have anderstud opencl cant understand anyother dimention than 1D tables(buffers).
So you must work around this!
If you now want 3d data to be accesed as a table, you can use something like that in your program:
uint x_index = get_global_id(0);
uint y_index = get_global_id(1);
uint z_index = get_global_id(2);
uint oneD_index = x_index + y_index * tables_x_dim + z_index * tables_x_dim * tables_y_dim;
For higher dimentional tablesalso opencl can't do something with compute units dimentions.
To recup you can use only 1D buffers or tables in practice in opencl's env. and at most 3 numbers for compute units dimentions.
The last limitation on this library is the fact that opencl has also limits on how many numbers you can put on each dimention.
Generally the maximum total number of compute units for this library is 16777216.
So you can use any table as long as it is 3d and lower in iorder or consists of less than 16777216 cells for the function get_global_id.
General  approach for heier dimention tables and with larger size, is to change your thinking about tables all together.
For example you can compute a huge table with size: 16777216 * 1024 if you just put a loop in your programm for every cu to compute not one but 1024 cells!
So the limitations of this is in your mind! Have fun and if you want drag farther this simple lib!
