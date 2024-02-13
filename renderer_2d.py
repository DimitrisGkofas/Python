from KERNEL import programs_ios_class
import numpy as np
import pygame

from kernels_code import render

progs = programs_ios_class()
progs.new_table('rend_table', (128, 128, 3), np.uint8)
progs.new_table('pos_ids_table', (128, 128), np.uint32)

screen_array = progs.table('rend_table')
table_pos = progs.table('pos_ids_table')
table_pos[1, 1] = np.uint8(255)

progs.new_program('rendering', render, (128, 128))

# Initialize Pygame
pygame.init()

# Set up Pygame window
window_dim = 720
screen = pygame.display.set_mode((window_dim, window_dim))
pygame.display.set_caption("OpenCL Rendering")

def render_output(screen):
    window = pygame.display.get_surface()
    # Convert to Pygame surface
    screen_surface = pygame.surfarray.make_surface(screen)
    # Get the dimensions of the Pygame window
    window_width, window_height = window.get_size()
    # Scale the surface to fill the window while maintaining the aspect ratio
    scaled_surface = pygame.transform.scale(screen_surface, (window_width, window_height))
    # Blit the surface onto the Pygame window
    window.blit(scaled_surface, (0, 0))

# Set up clock for tracking FPS
clock = pygame.time.Clock()

# Main Pygame loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # change one value in the pos_ids_table
    table_pos[1, 1] = np.uint8(np.random.randint(0, 255))
    # render in gpu the blocks
    progs.run_program('rendering', 'rend_table', 'pos_ids_table')
    # Render output screen_array
    render_output(screen_array)
    # Update the Pygame window
    pygame.display.flip()
    # Cap the frame rate to 60 FPS
    clock.tick(60)
    # Calculate FPS
    fps = clock.get_fps()
    # Print FPS to the console
    print(f"FPS: {fps:.2f}")

pygame.quit()