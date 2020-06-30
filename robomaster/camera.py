# python camera.py tello image'
# python camera.py robo image
# python camera.py video

import cv2
import time
import os
import argparse
import sys
import threading

parser = argparse.ArgumentParser()
parser.add_argument('robot', choices=['robo', 'tello'], default='robo', help='robo or tello')
parser.add_argument('mode', choices=['image', 'video'], default='image',help='image or video')
args = parser.parse_args()

# if args.mode not in ('image', 'video'):
# 	print('invalid mode, choose image or video')
# 	sys.exit()

if args.robot == 'robo':
	from EP_api import findrobotIP, Robot
	robot_ip = findrobotIP()
	# robot = Robot('192.168.2.1')
	robot = Robot(robot_ip)

	robot.startvideo()
	while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
		pass
	width = robot.cap.get(3)
	height = robot.cap.get(4)

elif args.robot == 'tello':
	from Tello_api import Tello
	robot = Tello()
	robot.startvideo()
	robot.startstates()
	while robot.frame is None: # this is for video warm up. when frame is received, this loop is exited.
		pass

def command_thread():
	while True:
		cmd_input = input('enter command: ')
		if cmd_input == 'exit':
			break
		else:
			robot._sendcommand(cmd_input)

commandthread = threading.Thread(target=command_thread, daemon=True)
commandthread.start()


if args.mode == 'video': 

	savedir = os.getcwd() + '/videos'
	count = 0

	if not os.path.exists(savedir):
		os.mkdir(savedir)
		print(f'Created {savedir}!')
		time.sleep(2)
	while os.path.exists(f'{savedir}/{count}.avi'):
		count += 1
	print(f'saving video to: {savedir}/{count}')

	out = cv2.VideoWriter(f'{savedir}/{count}.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30.0, (1280, 720)) # creates an empty video even if recording never started
	# out = cv2.VideoWriter(f'{savedir}/{count}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (1600, 900))
	print(f'Press spacebar to record video at:{savedir}/{count}.mp4')
	record = False

if args.mode == 'image':
	savedir = os.getcwd() + '/captures'
	count = 0 

	if not os.path.exists(savedir):
		os.mkdir(savedir)
		print(f'Created {savedir}!')
		time.sleep(2)


while True:
	if args.mode == 'video' and record:
		out.write(robot.frame)

	cv2.namedWindow('Live video', cv2.WINDOW_NORMAL)
	if args.robot == 'robo':
		cv2.imshow('Live video', robot.frame)
	if args.robot == 'tello':
		cv2.imshow('Live video', cv2.flip(robot.frame, 0))

	k = cv2.waitKey(16) & 0xFF
	if k == 27: # press esc to stop
		print("Quitting")
		if args.robot == 'robo':
			robot.exit()
		elif args.robot == 'tello':
			robot.exit()
		break
		
	elif k == 32: # spacebar
		if args.mode == 'video':
			record = not record
			print(f'Recording: {record}')
		elif args.mode == 'image':
			while os.path.exists(f'{savedir}/{count}.jpg'):
				count += 1
			print(f'saving images to: {savedir}/{count}.jpg')
			if args.robot == 'robo':
				cv2.imwrite(f'{savedir}/{count}.jpg', robot.frame)
			elif args.robot == 'tello':
				cv2.imwrite(f'{savedir}/{count}.jpg', cv2.flip(robot.frame, 0))

	# elif k != -1: # press wasdqe to control tello. u,j to control height. t,l to takeoff and land
	# 	# assert args.robot == 'tello'
	# 	robot.act(k) 

	# elif k == ord('w'):
	# 	robot._sendcommand('chassis move x 0.2')

	# elif k == ord('a'):
	# 	robot._sendcommand('chassis move y -0.2')

	# elif k == ord('s'):
	# 	robot._sendcommand('chassis move x -0.2')

	# elif k == ord('d'):
	# 	robot._sendcommand('chassis move y 0.2')
	# elif k == ord('q'):
	# 	robot._sendcommand('chassis move z -5')
	# elif k == ord('e'):
	# 	robot._sendcommand('chassis move z 5')

	# if k == ord('i'): # up and down arrow sometimes dont work
	# 	robot._sendcommand('gimbal move p 1')
	# if k == ord('k'): 
	# 	robot._sendcommand('gimbal move p -1')
	# if k == ord('j'): # up and down arrow sometimes dont work
	# 	robot._sendcommand('gimbal move y -1')
	# if k == ord('l'): 
	# 	robot._sendcommand('gimbal move y 1')

if args.mode == 'video':
	out.release()
cv2.destroyAllWindows()