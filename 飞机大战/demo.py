#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   demo.py
@Time    :   2023/11/05 12:41:59
@Author  :   不要葱姜蒜
@Version :   1.0
@Desc    :   None
'''

# coding: utf-8

import pygame
import random
import sys

# 初始化 Pygame
pygame.init()

# 游戏窗口的大小
WINDOW_WIDTH = 480
WINDOW_HEIGHT = 600

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# 创建游戏窗口
window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("飞机大战")

# 加载图片
player_img = pygame.image.load("player.svg").convert_alpha()
enemy_img = pygame.image.load("enemy.svg").convert_alpha()
bullet_img = pygame.image.load("bullet.svg").convert_alpha()

explosion_imgs = []
filename = "explosion.svg"
img = pygame.image.load(filename).convert_alpha()
explosion_imgs.append(img)

# 创建玩家飞机
player_width = 10
player_height = 20
player_x = (WINDOW_WIDTH - player_width) // 2
player_y = WINDOW_HEIGHT - player_height - 10
player_speed = 5
player = pygame.Rect(player_x, player_y, player_width, player_height)

# 创建敌机
enemy_width = 10
enemy_height = 10
enemy_speed = 3
enemies = []

# 创建子弹
bullet_width = 10
bullet_height = 20
bullet_speed = 10
bullets = []

# 创建爆炸效果
explosions = []

# 创建计分板
score = 0
font = pygame.font.SysFont(None, 30)

# 创建移动标记
moving_left = False
moving_right = False

# 游戏循环
clock = pygame.time.Clock()
while True:
    
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                moving_left = True
            elif event.key == pygame.K_RIGHT:
                moving_right = True
            elif event.key == pygame.K_SPACE:
                bullet_x = player.x + player_width // 2 - bullet_width // 2
                bullet_y = player.y - bullet_height
                bullet = pygame.Rect(bullet_x, bullet_y, bullet_width, bullet_height)
                bullets.append(bullet)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                moving_left = False
            elif event.key == pygame.K_RIGHT:
                moving_right = False

    # 移动玩家飞机
    if moving_left:
        player.x -= player_speed
    if moving_right:
        player.x += player_speed 
    if player.left < 0:
        player.left = 0
    elif player.right > WINDOW_WIDTH:
        player.right = WINDOW_WIDTH

    # 移动敌机
    if len(enemies) < 5:
        enemy_x = random.randint(0, WINDOW_WIDTH - enemy_width)
        enemy_y = random.randint(-500, -enemy_height)
        enemy = pygame.Rect(enemy_x, enemy_y, enemy_width, enemy_height)
        enemies.append(enemy)
    for enemy in enemies:
        enemy.y += enemy_speed
        if enemy.top > WINDOW_HEIGHT:
            enemies.remove(enemy)

    # 移动子弹
    for bullet in bullets:
        bullet.y -= bullet_speed
        if bullet.bottom < 0:
            bullets.remove(bullet)

    # 碰撞检测
    for enemy in enemies:
        if player.colliderect(enemy):
            print('boom. enemy: [', enemy.size, '=', enemy.x, enemy.y, ']', 'player: [', player.size, '=', player.x, player.y, ']')
            pygame.quit()
            sys.exit()
        for bullet in bullets:
            if bullet.colliderect(enemy):
                explosions.append(enemy)
                bullets.remove(bullet)
                enemies.remove(enemy)
                score += 10
                break

    # 绘制游戏界面
    window.fill(BLACK)
    window.blit(player_img, player)
    for enemy in enemies:
        window.blit(enemy_img, enemy)
    for bullet in bullets:
        window.blit(bullet_img, bullet)
    for explosion in explosions:
        window.blit(explosion_imgs[0], explosion)
        explosions.remove(explosion)
    score_text = font.render("Score: {}".format(score), True, WHITE)
    window.blit(score_text, (10, 10))
    pygame.display.update()

    # 控制游戏帧率
    clock.tick(60)