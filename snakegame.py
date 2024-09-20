## snakegame.py  -  Importable module for usage by the machine-learning training agent.
# This is the MACHINE PLAYABLE VERSION OF: snakegame-human-playable.py

import pygame
import random
from collections import namedtuple
from enum import Enum
import numpy as np


# ###########################################    GLOBAL INITIALIZATION    ##############################################


NAME_SYSTEM_FONT: str = 'arial'
SIZE_FONT: int = 25
COLOR_FONT: tuple = (255, 255, 255)  # White
COLOR_FOOD: tuple = (200, 0, 0)  # Red
COLOR_SNAKE_OUTER: tuple = (0, 0, 255)  # Medium Blue
COLOR_SNAKE_INNER: tuple = (0, 100, 255)  # Light Blue
COLOR_BACKGROUND: tuple = (0, 0, 0)  # Black
SIZE_GRID: int = 20  # Game grid interval (grid square side-length) in pixels
SPEED_GAME: int = 100  # Lower is slower. 5-10 is a good speed range for humans.
  # NOTE: The machine-playable training version runs at maximum speed to minimize training times.
  # For fast training (which is almost certainy what you want) make this number high, like 1000 or 10000.
  # TODO: Clarify the speed instructions. re-write these after researching how speed value works precisely.

Point = namedtuple('Point', 'x, y')
# Using a namedtuple here allows the pretty/convenient access such as: head.x, head.y


# #############################################    CLASS DEFINITIONS    ################################################

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
# Using a subclass of Enum like this allows pretty/convenient usage of needed integer values,
#   with semantically-helpful names, such as: Direction.UP instead of remembering that is the integer 3.
#   NOTE: See additional comments about his Enum after the _move() method below.


class SnakeGameML:
    def __init__(self, w=640, h=480):
        self.w: int = w
        self.h: int = h
        self.display: pygame.Surface = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()  # TODO: For type hint is this int?

        self.direction: Direction = Direction.RIGHT  # TODO: What all comprises the 'state' of the game?

        self.head: Point = Point(self.w / 2, self.h / 2)  # The snake starts out exactly in the center.

        # 'snake' attribute is a list of the snakes Point tuples with the head first (left-most) and to start out,
        # the snake has two additional body cells, laid out to the left. The snake starts out heading to the right.
        self.snake = [self.head,
                      Point(self.head.x - SIZE_GRID, self.head.y),
                      Point(self.head.x - (2 * SIZE_GRID), self.head.y)]

        self.score: int = 0
        self.food: Point = Point(0, 0)
        self._place_food()  # As part of init, call this private method to place food. Also called after each is eaten.
        self.frame_iteration: int = 0  # (**==--==** New var added for machine-playable version.)
        self.reset()  # (**==--==** Call to this new function added for machine-playable version.)
    # end def SnakeGameML.__init__()  -  #

    def reset(self):  # (**==--==** Function added for machine-playable version.)
        # INITIALIZE GAME STATE
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)  # The snake starts out exactly in the center.
        # 'snake' attribute is a list of the snakes Point tuples with the head first (left-most) and to start out,
        # the snake has two additional body cells, laid out to the left. The snake starts out heading to the right.
        self.snake = [self.head,
                      Point(self.head.x-SIZE_GRID, self.head.y),
                      Point(self.head.x-(2*SIZE_GRID), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
    # end def reset()  -  # (**==--==** Function added for machine-playable version.)

    def _place_food(self):
        x = random.randint(0, (self.w - SIZE_GRID) // SIZE_GRID) * SIZE_GRID
        y = random.randint(0, (self.h - SIZE_GRID) // SIZE_GRID) * SIZE_GRID
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
    # end def _place_food()  -  # // About floor division: https://www.geeksforgeeks.org/floor-division-in-python/

    def advance_game_step(self, action):  # (**==--==** New arg 'action' added for machine-playable version.)
        self.frame_iteration += 1  # (**==--==** New var increment added for machine-playable version.)
        for event in pygame.event.get():  # USER INPUT
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # (**==--==** Key-control user input disabled here for machine-playable version.)
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT:
            #         self.direction = Direction.LEFT
            #     elif event.key == pygame.K_RIGHT:
            #         self.direction = Direction.RIGHT
            #     elif event.key == pygame.K_UP:
            #         self.direction = Direction.UP
            #     elif event.key == pygame.K_DOWN:
            #         self.direction = Direction.DOWN
        # end event loop and user input  -  #

        # MOVE: Calls private _move() method to set the NEW HEAD based on DIRECTION, THEN WE PREPEND THIS NEW Point
        #   ONTO THE LEFT SIDE OF THE 'snake' LIST.

        # MACHINE-PLAYABLE INPUT/ACTION
        self._move(action)# (**==--==** Changed for machine-playable version.)
        self.snake.insert(0, self.head)

        # CHECK FOR COLLISION WITH SELF OR WALL AND END THE GAME IF THAT OCCURS. NOTE: A tuple of info is returned.
        reward = 0  # (**==--==** REWARD SYSTEM - Added for machine-playable version and model training.)
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score  # (**==--==** Added 'reward' return val for machine-playable version.)

        # CHECK FOR SAME POSITION AS FOOD (EATING IT) AND IF SO PLACE A NEW FOOD OR - JUST MOVE.
        # NOTICE THAT MOVING HAS BEEN MOSTLY TAKEN CARE OF ALREADY AND NOW WE JUST REMOVE THE TAIL END OF THE SNAKE
        # IF WE DID NOT HIT THE FOOD.
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            # IMPORTANT! When we eat food, WE GROW, so we do not discard the tail cell of the snake when we eat.
        else:
            self.snake.pop()  # Discards the tail cell of the snake since we have already set the new head position.

        # UPDATE UI AND CLOCK
        self._update_ui()
        self.clock.tick(SPEED_GAME)
        return reward, game_over, self.score  # (**==--==** Added 'reward' return val for machine-playable version.)
    # end def advance_game_step()  -  # This is like the contents of your main while loop. Think of the game class
    #   as being like moving all the code you have in global space in a simple pygame into this class, and then you
    #   have a public method which is like you MAIN WHILE LOOP. Note that, in our main() (below) is where we have the
    #   infinite while loop and from there we simply call this advance_game_step(). One more thing is to think of
    #   the __init__() of your game class as simply being the code you used to have in your global space from the start
    #   of the python file up to any if-name-main you had before moving the global code into a class like this.
    #   It's a clean/simple to move all the stuff you used to put at the global
    #   level, into such a game class, with the little nuance of this while loop sitting outside in main and doing
    #   little or nothing more than calling your advance_game_step() public method. I almost called this method
    #   advance_fram() or advance_game_frame().
    #   TODO: Can we call one frame a 'tick' ? Always rememver that in-between frames/game-steps there is a varying
    #     amount of time that passes, which is why we might us motion multipliers like delta_time (delta_time is
    #     a separate topic of its own) and so this info might help determine the most accurate definition and proper
    #     usage of 'tick' in the pygame context. I like my comments and docs to be well written and accurate.
    #     (similarly, I am very careful about how I name variables, classes, methods, functions, etc.)

    # (**==--==** Converted _is_collision to public method for machine-playable version. Was private for human-playable.)
    def is_collision(self, cell=None):  # (**==--==** Added new 'point' arg for machine-playable version, public method.)
        if cell is None:
            cell = self.head
        # COLLISION WITH DISPLAY BOUNDARY
        if cell.x > self.w - SIZE_GRID or cell.x < 0 or cell.y > self.h - SIZE_GRID or cell.y < 0:
            return True
        # COLLISION WITH SNAKE CELL
        if cell in self.snake[1:]:
            return True

        return False
    # end def _is_collision()  -  #

    def _update_ui(self):
        self.display.fill(COLOR_BACKGROUND)

        # PAINT ALL SNAKE CELLS - Two squares are painted for each cell, one inside the other. just to look nice.
        for cell in self.snake:
            pygame.draw.rect(self.display, COLOR_SNAKE_OUTER, pygame.Rect(cell.x, cell.y, SIZE_GRID, SIZE_GRID))
            pygame.draw.rect(self.display, COLOR_SNAKE_INNER, pygame.Rect(cell.x + 4, cell.y + 4, 12, 12))

        # PAINT THE FOOD CELL
        pygame.draw.rect(self.display, COLOR_FOOD, pygame.Rect(self.food.x, self.food.y, SIZE_GRID, SIZE_GRID))

        # PAINT THE SCORE IN THE UPPER LEFT
        text = font.render("Score: " + str(self.score), True, COLOR_FONT)
        self.display.blit(text, [0, 0])

        pygame.display.flip()
    # end def _update_ui()  -  #

    # (**==--==** Modified _is_collision for machine-playable version.)
    def _move(self, action):  # (**==--==** Arg 'direction' changed to 'action' for machine-playable version.)
        # (**==--==** BEGIN NEW DIRECTION DETERMINATION for machine-playable version.)
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        direction_index = clockwise.index(self.direction)  # TODO: Read up on this List.index() method. New? Rolled my own in years past.
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clockwise[direction_index]  # FORWARD
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (direction_index + 1) % 4
            new_direction = clockwise[next_index]  # RIGHT: r -> d -> l -> u
        else:  # [0, 0, 1]
            next_index = (direction_index - 1) % 4
            new_direction = clockwise[next_index]  # LEFT: r -> u -> l -> d
        self.direction = new_direction
        # (**==--==** END NEW DIRECTION DETERMINATION for machine-playable version.)

        x = self.head.x
        y = self.head.y
        # (**==--==** NOTE: Was local var 'direction'. Now references self.direction for machine-playable version.)
        if self.direction == Direction.RIGHT:
            x += SIZE_GRID
        elif self.direction == Direction.LEFT:
            x -= SIZE_GRID
        elif self.direction == Direction.DOWN:
            y += SIZE_GRID
        elif self.direction == Direction.UP:
            y -= SIZE_GRID

        self.head = Point(x, y)
    # end def _move()  -  # The Enum integer values of these directions are irrelevant and only part of how Enums work.
    #   In other words, the Enum integer values of 1,2,3,4 do not come into play in any calcualtions. They take effect
    #   right here and only determine (based on their semantic meaning of their labels) whether it is x or y which is
    #   added to or subtracted from.


# #############################################    FUNCTION DEFINITIONS    #############################################

# This program does not currently have any global-level functions. One of the features of this pygame code is that
# we have NOT put everything at the global level and do nearly everything within a 'game' class featuring a public
# game frame/step-advance method which is called from the global-resident main which has a simple main while loop.
# The point of these comments and section-marker is that in many programs, especially smaller ones, there could be
# global-level functions defined and if-so, this is my preferred and possibly the best location for those, after
# class definitions.

# Currently no global functions.


# ###############################################    INITIALIZATION    #################################################

# Things mostly start executing inside your if-name-main but some programs might have significant initialization
# before then OR possible, the code is importable, then this section could be substantial. There is usually some kind
# of setup/initialization that occurs before the 'main' code execution begins. Every program is different, but these
# major sections I have marked are almost universal and belong in the places I have indicated. I find these clearly
# visible section markers to be very helpful for really 'seeing' code as it might be scrolling by fast and quickly
# getting to the part of code you are looking for. In this simple program, we don't do very much initialization here.

pygame.init()
font: pygame.Font = pygame.font.SysFont(NAME_SYSTEM_FONT, SIZE_FONT)


# ################################################    INSTANTIATION    #################################################

# This program also does not instantiate class before the main execution gets going, BUT especially with PyGame, many
# games/programs WILL do possibly a lot of instantiation of class instances (commonly Sprite subclass instances) and
# this will likely be a significant section in larger programs. A common scenario is that configuration data about all
# the different things/objects/sprites/images/sounds/animations in a game or complex program would be parsed and
# processed to use game classes to create all the live instances of those classes as the game/program gets ready to
# start really running. This will come after initialization and before your main exection, so this is where this kind
# of code goes. You can use higher-level classes for this work to organize things or do it all in a game class or
# have global-level code here (and/or use global functions) BUT HERE is where you will or should likely be triggering
# and managing all that instantiation from. See my 'pygamefun' repo for a good example of this and some of my above
# section recommendations.

# This instantiation activity is likely a place where some global varaibles/objects might be populated and there is
# a high chance that you will want to use the SAME NAMES inside inner scoprs for arguments and more, SO there is
# almost guaranteed shadowing of variables likely to occur and that is likely to be in this section near the top.
# Hence, some important recommendations to help you avoid some potentially nasty bugs:

# AVOID SERIOUS BUGS FROM VARIABLE NAME SHADOWING, with additional benefits:
# Naming convention for variables at the global level which may have shadowing issues becuase their name
# is one likely to also be used in inner scopes. This is a system of a few prefixes to use on variable names, with
# a rough meaning:
# gr_    Lives in global scope and is intended for READ ONLY. Some bugs are still possible.
# gw_    Lives in global scope and may be written to. Many kinds of bugs possible if care is not taken.

# Currently no global-level instantation prior to if-name-main.


# ###############################################    MAIN EXECUTION    #################################################

# NO MAIN BLOCK IN MACHINE-PLAYABLE, ML TRAINING VERSION

# (**==--==** NOTE: Our minimal if-name-main code removed for machine-playable version. This code must be imported.)
if __name__ == '__main__':
    print("Machine-playable version. This must be imported by the agent for playing. Do not execute directly.")
    exit(1)
# end if __name__ main  -  #


##
#


# ###################################################    NOTES    ######################################################

# PYGAME-CE DOCS:
# https://pyga.me/docs/

# NOTE: gr_ prefix on variable gr_game_over fixes a shadowing warning, and it indicates that this global variable can
#  be expected to ONLY be READ from (hence the r - gr = globally read). If I have such a global variable (or data
# structure like a global cache perhaps) and I know that some downstream/child in some more inner scope CAN and MAY
# or WILL write, update, initialize or set any value or component value at any time, THEN I may use the gw_ prefix
# meaning 'globally written'. I primarily use these prefixes for solving shadowing/scope warnings. It works well
# for me. Sometimes you still want things in global space. There is nothing wrong with it for certain variables and
# design patterns. For sure, leaving variables shadowed CAN and WILL lead to bad bugs, so to mitigate challenges of
# havings things in global space, at least address shadow warnings and use prefixes like this or a similar method.
# This is a very valuable strategy. Of course, you have most stuff in inner scopes, but you still want some things
# global sometimes, so have a strategy for it.


##
#
