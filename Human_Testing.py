import gym
import pygame
import numpy as np

def human_play():
    env = gym.make('CartPole-v1', render_mode="human")
    observation, _ = env.reset()
    
    done = False
    total_reward = 0
    
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    clock = pygame.time.Clock()
    
    print("Use 'A' to move left, 'D' to move right, and 'Q' to quit.")
    
    while not done:
        env.render()
        
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    action = 0  # Move cart left
                elif event.key == pygame.K_d:
                    action = 1  # Move cart right
                elif event.key == pygame.K_q:
                    done = True
        
        if action is not None:
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
        
        clock.tick(60)
    
    print(f"Game Over! Your score: {total_reward}")
    env.close()
    pygame.quit()
    return total_reward

if __name__ == "__main__":
    print("Let's Play CartPole!")
    print("Controls: 'A' - move left, 'D' - move right, 'Q' - quit")
    human_score = human_play()
    print(f"Your final score: {human_score}")