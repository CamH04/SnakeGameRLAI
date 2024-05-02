import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

'''
pygame: The core module for creating games with Pygame. Handles graphics, sound, events, etc.
random: Used for generating random positions for food placement.
enum import Enum: Allows to define the Direction class for representing movement directions in a clear, organized way.
collections import namedtuple: Creates the Point namedtuple, which is a simple and efficient way to store x and y coordinates.
numpy as np: Used for array manipulations when working with the snake actions.
'''

pygame.init() # Initializes all the necessary Pygame modules 
#font = pygame.font.Font('arial.ttf', 25) 
font = pygame.font.SysFont('arial', 25)
'''
Loads a font named 'arial.ttf' (if found) with size 25, or uses a system font named 'arial'
This will be used to display the score
'''

class Direction(Enum): # Defines an enumeration for the four possible directions which improves code readability and makes it easier to manage movement
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y') # Creates a simple data structure to hold x and y coordinates, making it easier to represent positions of the snake's head and the food

# Defines colors (in RGB format) for elements in the game (background, snake, food)
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20 # The size of each snake block and the food item in pixels 
SPEED = 40 # Controls the speed of the game. It's likely related to the clock tick rate

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
        '''
        Constructor: Initializes the game environment.
        Sets the width (self.w) and height (self.h) of the game window.
        Initializes Pygame using pygame.display.set_mode() and sets the window caption.
        Creates a clock object (self.clock) to control the game speed.
        Calls self.reset() to begin a new game.
        '''


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        '''
        Sets the snake's direction to Direction.RIGHT.
        Positions the snake's head in the center of the screen and creates the initial body segments.
        Resets the score to zero.
        Places the first food item using self._place_food().
        Resets the frame counter.
        '''

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
            '''
            Calculates random x and y coordinates within the game boundaries, aligned to the block size.
            Ensures the food isn't placed on top of the snake (calls itself recursively if it does).
            '''

    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
        '''
        Handles a single frame of the game:
        Increments the frame counter.
        Handles User Input: Checks for events (like keyboard presses or the quit button).
        Moves the Snake: Calls self._move(action) to update the snake's position based on the provided action.
        Checks for Game Over: Calls self.is_collision() and checks the frame counter. If a game over condition is met, sets game_over to True and returns an appropriate reward (-10).
        Places New Food (if needed): If the snake head is at the food's position, increments the score, places new food, and returns a reward (10). Otherwise, just removes the tail segment.
        Updates the Display: Calls self._update_ui() to redraw the screen.
        Controls Frame Rate: Uses self.clock.tick(SPEED) to maintain the speed.
        Returns: reward, game_over, and score.
        '''

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False
        '''
        Accepts an optional pt argument (a Point object), and defaults to checking the snake's head.
        Returns True if the point is out of bounds or if the point is part of the snake's body (excluding the head).
        Returns False otherwise.
        '''


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        '''
        Redraws the game screen:
        Fills the background with black.
        Draws the snake using blue rectangles.
        Draws the food as a red rectangle.
        Renders the score text.
        Updates the display using pygame.display.flip().
        '''

    def _move(self, action):
        # [straight, right, left]
        
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir
        '''
        pdates the snake's direction based on the action:
        Defines clock_wise for easy direction changes.
        Interprets the action array (likely [straight, right, left]) to determine the new direction.
        Calculates the new head position (x, y) based on the snake's updated direction.
        '''

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)