import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Set up window dimensions
width, height = 768, 768

# Create the Pygame window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simple Pygame Window")

# Initial array
screen_np = np.zeros(shape=(128, 128, 3), dtype=np.uint8)



# Game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update game logic here
    # Convert to Pygame surface
    screen_surface = pygame.surfarray.make_surface(screen_np)
    # Scale the surface to fill the window while maintaining the aspect ratio
    scaled_surface = pygame.transform.scale(screen_surface, (768, 768))
    # Blit the surface onto the Pygame window
    screen.blit(scaled_surface, (0, 0))
    # Draw game elements here

    # Update the display
    pygame.display.flip()