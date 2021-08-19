from riglib.stereo_opengl.window import *
from riglib.stereo_opengl.models import *

import pygame


target_colors = {
    "red": (1,0,0,0.75),
    "yellow": (1,1,0,0.75),
    "green":(0., 1., 0., 0.75),
    "blue":(0.,0.,1.,0.75),
    "magenta": (1,0,1,0.75),
    "pink": (1,0.5,1,0.75),
    "purple":(0.608,0.188,1,0.75),
    "teal":(0,0.502,0.502,0.75),
    "olive":(0.420,0.557,0.137,.75),
    "orange": (1,0.502,0.,0.75),
    "hotpink":(1,0.0,0.606,.75),
    "elephant":(0.5,0.5,0.5,0.5),
}

wd  = Window()
# wd.screen_init()
# wd._get_renderer()
# """exp = WindowWithExperimenterDisplay(wd) # This is the second display
# exp._get_renderer()"""

# wd.screen_dist = 60 #in whatevet unit we don't know yet
# wait_time = 10000 # in ms

# x = 0
# y = 0
# os.environ['SDL_VIDEO_WINDOW_POS'] =  "%d,%d" % (x,y)
# os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"

# wd.screen_init()
# wd.set_eye((0, -wd.screen_dist, 0), (0,0))

# #put a sphere at the center of the field.

# sp1 = Sphere(5)
# sp1.translate(0,-20,200) 
# wd.add_model(sp1)
# sp1.color = (1,1,0,0.25)

# sp2 = Sphere(5)
# sp2.translate(0,0,200) 
# wd.add_model(sp2)
# sp2.color = (1,1,0,0.5)


# sp3 = Sphere(5)
# sp3.translate(0,20,200) 
# sp3.color = (1,1,0, 0.75)
# wd.add_model(sp3)

# sp4 = Sphere(5)
# sp4.translate(0,40,200) 
# sp4.color = (1,1,0, 0.75)
# wd.add_model(sp4)

# #draw the initial world
# wd.requeue()
# wd.draw_world()
# pygame.time.wait(2000) #check every 100 ms


# print("Press any key to exit pygame window...")
# n=0
# while True:
#     for event in pygame.event.get():
# 	    if event.type == pygame.KEYDOWN:
#         	pygame.quit()
#         	quit()
#     #refresh the models
#     pygame.time.wait(15000) 
#     sp1.color = (1,1,0,0.25)

#     # refresh the graphics
#     wd.requeue()
#     wd.draw_world()

#     # pygame.time.wait(15000) 
#     # sp1.color = (1,0,0,0.5)

#     # # refresh the graphics
#     # wd.requeue()
#     # wd.draw_world()

#     # pygame.time.wait(15000)
#     # sp1.color = (1,0,0,0.75)

#     # # refresh the graphics
#     # wd.requeue()
#     # wd.draw_world()

#     # pygame.time.wait(15000)
#     # sp1.color = (1,0, 0,1)

#     # refresh the graphics
#     wd.requeue()
#     wd.draw_world()
#     pygame.time.wait(1000) #check every 10 s
#     n += 1
#     if n == 18 :
# 	    pygame.time.wait(1000)
# 	    break
	
# print ("It works")



wd_2d  = Window2D()
wd_2d.screen_init()
wd_2d._get_renderer()
"""exp = WindowWithExperimenterDisplay(wd) # This is the second display
exp._get_renderer()"""
wd_2d.screen_cm = [16800,10600]
print(wd_2d.screen_cm)
wd_2d.screen_dist = 60 #in whatevet unit we don't know yet
wait_time = 10000 # in ms

x = 0
y = 0
os.environ['SDL_VIDEO_WINDOW_POS'] =  "%d,%d" % (x,y)
os.environ['SDL_VIDEO_X11_WMCLASS'] = "monkey_experiment"

wd_2d.screen_init()
# wd_2d.set_eye((0, -wd.screen_dist, 0), (0,0))

#put a sphere at the center of the field.

sp1 = Sphere(500, color=target_colors['yellow'])
sp1.translate(0,-20,0) 
wd_2d.add_model(sp1)
# sp1.color = (1,1,0,0.75)

# sp2 = Sphere(5)
# sp2.translate(0,0,200) 
# wd.add_model(sp2)
# sp2.color = (1,1,0,0.5)


# sp3 = Sphere(5)
# sp3.translate(0,20,200) 
# sp3.color = (1,1,0, 0.75)
# wd.add_model(sp3)

# sp4 = Sphere(5)
# sp4.translate(0,40,200) 
# sp4.color = (1,1,0, 0.75)
# wd.add_model(sp4)

#draw the initial world
wd_2d.requeue()
wd_2d.draw_world()
pygame.time.wait(200) #check every 100 ms


print("Press any key to exit pygame window...")
n=0
while True:
    for event in pygame.event.get():
	    if event.type == pygame.KEYDOWN:
        	pygame.quit()
        	quit()
    #refresh the models
    sp1.color = (1,1,0,1)

    # refresh the graphics
    wd_2d.requeue()
    pygame.time.wait(1000) 
    wd_2d.draw_world()

    pygame.time.wait(2000) 
    sp1.color = (1,1,0,0)

#     # pygame.time.wait(15000) 
#     # sp1.color = (1,0,0,0.5)

#     # # refresh the graphics
#     # wd.requeue()
#     # wd.draw_world()

#     # pygame.time.wait(15000)
#     # sp1.color = (1,0,0,0.75)

#     # # refresh the graphics
#     # wd.requeue()
#     # wd.draw_world()

#     # pygame.time.wait(15000)
#     # sp1.color = (1,0, 0,1)

    # refresh the graphics
    wd_2d.requeue()
    wd_2d.draw_world()
    pygame.time.wait(2000) 
    sp1.color = (1,1,0,0)

    wd_2d.requeue()
    wd_2d.draw_world()
    pygame.time.wait(2000) 
    # pygame.time.wait(8000) #check every 8 s
    # n += 1
    # if n == 18 :
	#     pygame.time.wait(1000)
	#     break