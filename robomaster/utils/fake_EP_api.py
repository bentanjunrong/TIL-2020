import socket
from threading import Thread

class Robot():
	def __init__(self):
		print('Fake Robot initialized!')
        
	
	def _sendcommand(self, cmdstring, echo=True): 
		print('Command sent: {}'.format(cmdstring))
		return

	# def startvideo(self):
		

	def _receive_video_thread(self):
		"""
		Listens for video streaming (raw h264).
		Runs as a thread, sets self.frame to the most recent frame captured. frame processing and imshow should be in main loop
		"""
		


# ROBOT COMMANDS
	def move(self, inputstring):
		'''
		input: x dist y dist z angle vxy m/s Vz degree/s
		moves chassis by distance
		e.g. move('x 0.5') to move forward by 0.5m
		'''
		command = 'chassis move ' + inputstring
		return self._sendcommand(command)

	def rotate(self, inputstring):
		'''
		input: string in degrees
		rotate CW
		'''
		command = 'chassis move z ' + inputstring
		return self._sendcommand(command)

	def right(self, inputstring):
		'''
		input: integer in m/s
		constantly strafe right
		'''
		command = 'chassis speed y ' + inputstring
		return self._sendcommand(command)

	def forward(self, inputstring):
		'''
		input: integer in m per s
		constantly move forward
		'''
		command = 'chassis speed x ' + inputstring
		return self._sendcommand(command)

	def movearm(self, inputstring):
		"""
		input: x {val} y {val}
		"""
		command = 'robotic_arm move ' + inputstring
		return self._sendcommand(command)
		
	def openarm(self):
		return self._sendcommand('robotic_gripper open 1')

	def closearm(self):
		return self._sendcommand('robotic_gripper close 1')

	def rotategimbal(self, inputstring):
		'''
		relative to current position
		p angle y angle vp speed vy speed

		'''
		command = 'gimbal move ' + inputstring
		return self._sendcommand(command)

	def rotategimbalto(self, inputstring):
		'''
		relative to initial position
		p angle y angle vp speed vy speed
		'''
		command = 'gimbal moveto ' + inputstring
		return self._sendcommand(command)

	def stop(self):
		self._sendcommand('chassis move x 0')
		self._sendcommand('gimbal speed p 0 y 0') # stops if in speed. gimbal move will finish
		self._sendcommand('robotic_arm stop') 

	def center(self):
		self._sendcommand('gimbal moveto p 0 y 0')
		self._sendcommand('robotic_arm recenter') 

	def exit(self): # call this to stop robot and bring back to center position before exiting
		
		print('All done')