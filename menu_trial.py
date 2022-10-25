import pygame
import pygame_menu
from main import main_function

import numpy as np

# initializing the constructor
pygame.init()

# screen resolution
res = (1280, 720)
mytheme = pygame_menu.themes.THEME_ORANGE.copy()
mytheme.title_background_color=(0, 0, 0)
myimage = pygame_menu.baseimage.BaseImage(
    image_path='background.jpg',
    drawing_mode=pygame_menu.baseimage.IMAGE_MODE_FILL,
)
mytheme.background_color = myimage
# opens up a window
screen = pygame.display.set_mode(res)
def get_url_and_start():
    url = url_camera_phone.get_value()
    start('CAMERA_PHONE',url)



def start(device,url):
    main_function(device,url)


# Camera_Phone menu
phone_menu = pygame_menu.Menu('Camera Phone', 1280, 720,
                    theme=mytheme)
url_camera_phone = phone_menu.add.text_input('Enter the url: ',default='http://10.1.18.99:4747/video')
phone_menu.add.button('Start',get_url_and_start)
# second menu
second_menu = pygame_menu.Menu('Select device', 1280, 720,
                    theme=mytheme)

second_menu.add.button('Camera_Laptop',start,'CAMERA_LAPTOP','')
second_menu.add.button('Camera_Phone',phone_menu)
second_menu.add.button('Video',start,'VIDEO','')
second_menu.add.button('Exit', pygame_menu.events.EXIT)


# main menu
menu = pygame_menu.Menu('Fire detection', 1280, 720,
                        theme=mytheme)

menu.add.text_input('Enter your name: ', default='')
menu.add.button('Next',second_menu)
menu.add.button('Exit',pygame_menu.events.EXIT)
menu.mainloop(screen)