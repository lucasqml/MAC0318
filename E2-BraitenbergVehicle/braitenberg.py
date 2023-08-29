# MAC0318 Programação de Robôs Móveis
#
# Preencha o cabeçalho abaixo com seus dados. 
# Se o trabalho for feito em grupo, coloque os nomes de 
# todos os integrantes (copie e cole as linhas abaixo)
#
# Nome: Lucas Quaresma Medina Lam  
# NUSP: 11796399
#
# Nome: Luã Nowacki Scavacini Santilli
# NUSP: 11795492
#
# ---
#
# Atividade - Veículo de Braitenberg
#
# Objetivo:
#  - Implementar um agente reativo com o comportamento enamorado.
#
# Não se esqueça de executar este arquivo do diretório raiz do Duckievillage (p.ex.):
#   cd ~/MAC0318/duckievillage
#   conda activate duckietown
#   python3 assignments/braitenberg/braitenberg.py
#
# Instruções:
#  0. Preencha o cabeçalho acima
#  1. Assegure-se que o código submetido é o correto e funcionando. Se seu código não roda, você receberá nota 0 na atividade.
#  2. Submeta este arquivo no e-disciplinas.
#  3. Coloque esse arquivo no seu repositório pessoal (sugestão: fazendo um `fork` do repositório no gitlab). Você usará sua solução como base em outras atividades futuras.

import sys
import pyglet
import numpy as np
from pyglet.window import key
from duckievillage import create_env
import cv2

class Agent:
    # Agent initialization
    def __init__(self, environment):
        """ Initializes agent """
        self.env = environment
        # Color segmentation hyperspace
        # self.lower_hsv = np.array([5, 70, 90])  
        # self.upper_hsv = np.array([40, 255, 255])
        self.lower_hsv = np.array([20, 120, 120]) # Modificar
        self.upper_hsv = np.array([30, 255, 255]) # Modificar
        # Acquire image for initializing activation matrices
        img = self.env.front()
        img_shape = img.shape[0], img.shape[1]
        self.left_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        self.right_motor_matrix = np.zeros(shape=img_shape, dtype="float32")
        # TODO! Replace with your code
        # Each motor activation matrix specifies how much power is given to the respective motor after the image processing routines are applied
        self.left_motor_matrix[:, img_shape[1]//2:] = -1   # -1 'inhibits' motor, +1 'stimulates' motor
        self.right_motor_matrix[:, :img_shape[1]//2] = -1  # this implements the aggressive behaviour

    # Image processing routine - Color segmentation
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """ Returns a 2D array mask color segmentation of the image """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)//255
        #     masked = cv2.bitwise_and(image, image, mask=mask)
        return mask

    def send_commands(self, dt):
        ''' Agent control loop '''
        # acquire front camera image
        img = self.env.front()
        # run image processing routines
        P = self.preprocess(img)
        # build left and right signals
        L = float(np.sum(P * self.left_motor_matrix))
        R = float(np.sum(P * self.right_motor_matrix))
        limit = img.shape[0]*img.shape[1]
        # These are big numbers, thus rescale them to unit interval
        L = rescale(L, 0, limit)
        R = rescale(R, 0, limit)
        # Tweak with the constants below to get to change velocity or stabilize movements
        # Recall that pwm sets wheel torque, and is capped to be in [-1,1]
        gain = 5.0
        const = 0.2 # power under null activation - this ensures the robot does not halt
        pwm_left = const + R * gain
        pwm_right = const + L * gain
        # print('>', L, R, pwm_left, pwm_right) # uncomment for debugging
        # Now send command
        self.env.step(pwm_left, pwm_right)
        self.env.render('human')


def rescale(x: float, L: float, U: float):
    ''' Map scalar x in interval [L, U] to interval [0, 1]. '''
    return (x - L) / (U - L)


def main():
    print("MAC0318 - Assignment 1")
    env = create_env(
      raw_motor_input = True,
      seed = 101,
      map_name = './maps/nothing.yaml',
      draw_curve = False,
      draw_bbox = False,
      domain_rand = False,
      #color_sky = [0, 0, 0],
      user_tile_start = (0, 0),
      distortion = False,
      top_down = False,
      cam_height = 10,
      is_external_map = True,
      enable_lightsensor = False
    )
    for o in env.objects: o.scale = 0.085

    angle = env.unwrapped.cam_angle[0]

    env.start_pose = [[0.5, 0, 0.5], 150]
    env.reset()
    env.render('human')

    @env.unwrapped.window.event
    def on_key_press(symbol, modifiers):
        if symbol == key.ESCAPE: # exit simulation
            env.close()
            sys.exit(0)
        elif symbol == key.BACKSPACE or symbol == key.SLASH: # reset simulation
            print("RESET")
            env.reset()
        elif symbol == key.RETURN:  # Take a screenshot
            print('saving screenshot')
            img = env.render('rgb_array')
            cv2.imwrite(f'screenshot-{env.unwrapped.step_count}.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        env.render()

    # Instantiate agent
    agent = Agent(env)
    # Call send_commands function from periodically (to simulate processing latency)
    pyglet.clock.schedule_interval(agent.send_commands, 1.0 / env.unwrapped.frame_rate)
    # Now run simulation forever (or until ESC is pressed)
    pyglet.app.run()
    env.close()

if __name__ == '__main__':
    main()
