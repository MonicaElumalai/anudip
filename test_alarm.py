import pygame

pygame.mixer.init()
pygame.mixer.music.load("static/alarm.wav")
pygame.mixer.music.play()

input("Press Enter to stop")
