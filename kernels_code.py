kernel_one = """(float *positions, float coe) {
    uint index = get_global_id(0);
    positions[index] = positions[index] * coe;
}
"""