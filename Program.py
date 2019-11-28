#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:31:28 2019

@author: cis
"""
import random
from datetime import datetime
import subprocess as sp
import numpy as np
import time
import threading
import sys

int_var = 10
char_text1 = "abcdef"
char_text2 = "ghijk"

char_combined = char_text1 + char_text2
print(char_combined)

int_variable = random.randint(4, 10)

for i in range(1, 11):
    print(i+"\n")
    
char_variable = input("Please enter text here!")
print(char_variable)

now =   datetime.now()
char_variable = now.strftime("%H:%M:%S")
print(char_variable)

char_variable = datetime.today().strftime("%d.%m.%Y")
print(char_variable)

input_variable = int(input("Enter a number"))

if input_variable == 8:
    print("Equal")
elif input_variable > 8:
    print("bigger")
else:
    print("Smaller")
    
print("abc")
sp.call('clear',shell=True)

print("efg")

with open("My file.txt", "w") as f:
    char_variable = "my text"
    f.write(char_variable)
    
with open("My file.txt", "r") as r:
    while True:
        char_variable = f.read(1)
        if not char_variable:
            print("Reading Complete")
            break

print(char_variable)
    
char_variable = np.array([["FIRST", "SECOND", "THIRD"],
                 ["RED", "or", "BLUE"]])

for i in np.nditer(char_variable):
    print(i, end=" ")
    
print("ABC")
time.sleep(3)
print("DEFGH")

print('\x1b[0;31;46m' + 'ABC' + '\x1b[0;36;40m')

with open("My File.txt", "w") as f:
    char_variable = "line1 \n line2 \n line3"
    f.write(char_variable)

with open("My File.txt", "r") as f:
    var_line1, var_line2, var_line3 = f.readlines()
    
print(var_line1, var_line2, var_line3)

#
#def start_timer(arg):
#    time_start = time.time()
#    seconds = 0
#    minutes = 0
#    hours = 0
#    
#    while arg == "start":
#        try:
#            sys.stdout.write("\r{hours} Hours {minutes} Minutes {seconds} Seconds"
#                             .format(hours=hours, minutes=minutes, seconds=seconds))
#            sys.stdout.flush()
#            time.sleep(1)
#            seconds = int(time.time() - time_start) - minutes * 60
#            if seconds >= 60:
#                minutes += 1
#                seconds = 0
#            if minutes >= 60:
#                hours += 1
#                minutes = 0
#        except KeyboardInterrupt as e:
#            break
#
#
#enter = input("Press Enter to start the timer")
##start_timer()
#start_thread = threading.Thread(target=start_timer, args=("start",))
#start_thread.start()
#key = input("Press enter to stop")
#stop_thread = threading.Thread(target=start_timer, args=("stop",))
#stop_thread.start()

#if key == "":
#    break

var_integer = 5
var_char = "abc"
var_character = "234"
var_character_resoult = ""

var_character_resoult = str(var_integer + int(var_character))
print(var_character_resoult)

import subprocess
import os

p = subprocess.Popen(os.path.join("/usr", "bin", "vlc"))

import pygame

pygame.mixer.init()
key = input("Enter a to play a.mp3 and b to play b.mp3")
if key == "a":
    pygame.mixer.music.load("MUSIC/a.mp3")
elif key == "b":
    pygame.mixer.music.load("MUSIC/b.mp3")
else:
    pass

pygame.mixer.music.play()
while pygame.mixer.music.get_busy() == True:
    continue

    
from subprocess import Popen, PIPE
import time
import pyautogui

pyautogui.hotkey('f5')

import os
directory = "MEDIA"
if not os.path.exists(directory):
    os.makedirs(directory)
    
class_NAMES = []
count = 0
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):
        count +=1
        
