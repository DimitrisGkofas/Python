kernel_one = """(float *positions, float coe) {
    uint index = get_global_id(0);
    positions[index] = positions[index] * coe;
}
"""
render = """(uchar *screen, uchar *blocks) {
    uint xi = get_global_id(0);
    uint yi = get_global_id(1);

    uint index = xi + yi * 128;

    screen[index * 3] = blocks[index];
    screen[index * 3 + 1] = blocks[index];
    screen[index * 3 + 2] = blocks[index];
}
"""
