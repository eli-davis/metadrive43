# B"H

# based on openpilot 0.9.9
# -> openpilot/tools/sim/lib/keyboard_ctrl.py


import os
import time
import sys

import termios

from termios import (BRKINT, CS8, CSIZE, ECHO, ICANON, ICRNL, IEXTEN, INPCK,
                     ISTRIP, IXON, PARENB, VMIN, VTIME)


# Indexes for termios list.
IFLAG = 0
OFLAG = 1
CFLAG = 2
LFLAG = 3
ISPEED = 4
OSPEED = 5
CC = 6

'''
KEYBOARD_HELP = """
  | key  |   functionality       |
  |------|-----------------------|
  |  1   | Cruise Resume / Accel |
  |  2   | Cruise Set    / Decel |
  |  3   | Cruise Cancel         |
  |  r   | Reset Simulation      |
  |  i   | Toggle Ignition       |
  |  q   | Exit all              |
  | wasd | Control manually      |
"""
'''

KEYBOARD_HELP = """
| key    |   functionality       |
|--------|-----------------------|
|  UP    | Accel                 |
|  DOWN  | Brake                 |
|  LEFT  | Steer                 |
|  RIGHT | Brake                 |
|  x     | Exit all              |
"""

def getch() -> str:
    STDIN_FD = sys.stdin.fileno()
    old_settings = termios.tcgetattr(STDIN_FD)
    try:
        # set
        mode = old_settings.copy()
        mode[IFLAG] &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON)
        #mode[OFLAG] &= ~(OPOST)
        mode[CFLAG] &= ~(CSIZE | PARENB)
        mode[CFLAG] |= CS8
        mode[LFLAG] &= ~(ECHO | ICANON | IEXTEN)

        # enable CTRL-C
        mode[LFLAG] |= termios.ISIG

        mode[CC][VMIN] = 1
        mode[CC][VTIME] = 0
        termios.tcsetattr(STDIN_FD, termios.TCSAFLUSH, mode)



        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(STDIN_FD, termios.TCSADRAIN, old_settings)
    return ch

def print_keyboard_help():
    print(f"Keyboard Commands:\n{KEYBOARD_HELP}")

def get_char():
    c = getch()
    #print(c)

    if c == '\x1b':  # start of an escape sequence
        seq = sys.stdin.read(2)  # read the next two chars
        if seq == '[A':
            print("UP")
            return "UP"

        elif seq == '[B':
             print("DOWN")
             return "DOWN"

        elif seq == '[C':
             print("RIGHT")
             return "RIGHT"

        elif seq == '[D':
             print("LEFT")
             return "LEFT"
        else:
             print("ESC")
             return None
    else:
       print(f"Key: {c}")
       if c == 'x':
           os._exit(0)
       return None

def keyboard_poll():
    print_keyboard_help()

    while True:
        c = get_char()

    '''
    if c == '1':
      q.put(control_cmd_gen("cruise_up"))
    elif c == '2':
      q.put(control_cmd_gen("cruise_down"))
    elif c == '3':
      q.put(control_cmd_gen("cruise_cancel"))
    elif c == 'w':
      q.put(control_cmd_gen(f"throttle_{1.0}"))
    elif c == 'a':
      q.put(control_cmd_gen(f"steer_{-0.15}"))
    elif c == 's':
      q.put(control_cmd_gen(f"brake_{1.0}"))
    elif c == 'd':
      q.put(control_cmd_gen(f"steer_{0.15}"))
    elif c == 'z':
      q.put(control_cmd_gen("blinker_left"))
    elif c == 'x':
      q.put(control_cmd_gen("blinker_right"))
    elif c == 'i':
      q.put(control_cmd_gen("ignition"))
    elif c == 'r':
      q.put(control_cmd_gen("reset"))
    elif c == 'q':
      q.put(control_cmd_gen("quit"))
      break
    else:
      print_keyboard_help()
    '''


if __name__ == '__main__':
    keyboard_poll()
