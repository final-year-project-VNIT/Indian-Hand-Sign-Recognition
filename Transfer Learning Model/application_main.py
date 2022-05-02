# Importing Libraries

import numpy as np

import cv2
import os, sys
import time
import operator

from tkinter import Frame
from string import ascii_uppercase

import tkinter as tk
from PIL import Image, ImageTk

#from cyhunspell import Hunspell
import enchant

from keras.models import model_from_json
from keras.models import load_model
#import imagePreprocessingUtils as ipu

os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"

#gestures = ipu.get_all_gestures()
#cv2.imwrite("all_gestures.jpg", gestures)
img=cv2.imread("all_gestures.jpg")
#cv2.imshow("All_gestures", img)

#Application :
print(ascii_uppercase)
alpha_char="123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

class Application:

	def __init__(self):

#		self.hs = Hunspell('en_US')
		self.vs = cv2.VideoCapture(0)
		self.current_image = None
		self.current_image2 = None
	#	self.json_file = open("Models\model-bw.json", "r")
	#	self.model_json = self.json_file.read()
	#	self.json_file.close()

	#	self.loaded_model = model_from_json(self.model_json)
	#	self.loaded_model.load_weights("model_vgg16.h5")
		self.loaded_model = load_model("E:\\8th sem\\project\\model_vgg16.h5")  #path of vgg model

		# self.json_file_dru = open("Models\model-dru.json" , "r")
		# self.model_json_dru = self.json_file_dru.read()
		# self.json_file_dru.close()

		# self.loaded_model_dru = model_from_json(self.model_json_dru)
		# self.loaded_model_dru.load_weights("Models\model-dru.h5")

		# self.json_file_kdi = open("Models\model-kdi.json" , "r")
		# self.model_json_kdi = self.json_file_kdi.read()
		# self.json_file_kdi.close()
		# self.loaded_model_kdi = model_from_json(self.model_json_kdi)
		# self.loaded_model_kdi.load_weights("Models\model-kdi.h5")

		# self.json_file_smn = open("Models\model-smn.json" , "r")
		# self.model_json_smn = self.json_file_smn.read()
		# self.json_file_smn.close()

		# self.loaded_model_smn = model_from_json(self.model_json_smn)
		# self.loaded_model_smn.load_weights("Models\model-smn.h5")

		self.count = {}
		self.count['blank'] = 0
		self.blank_flag = 0

		#for i in ascii_uppercase:
		for i in alpha_char:
			self.count[i] = 0
		
		print("Starting the Interface")

		self.root = tk.Tk()
		frame1 = Frame(self.root, highlightbackground="blue", highlightthickness=5,width=800, height=800, bd= 0)
		frame1.pack()

		self.root.title("iSL Recognition")
		self.root.protocol('WM_DELETE_WINDOW', self.destructor)
		self.root.geometry("800x650")

		self.window = tk.Label(self.root)
		self.window.place(x = 100, y = 10, width = 580, height = 580)
		print("81")
		frame2 = Frame(self.root, highlightbackground="red", highlightthickness=2, width=275, height=275, bd= 0)
		frame2.pack(padx=700, pady=65)
		self.window2 = tk.Label(self.root, borderwidth=3, relief="solid") # initialize image panel
		self.window2.place(x = 400, y = 65, width = 275, height = 275)

		self.text = tk.Label(self.root)
		self.text.place(x = 100, y = 5)
		self.text.config(text = "ISL Recognition", font = ("Arial", 30, "bold"))
		print("line no. 88")
		self.window3 = tk.Label(self.root) # Current Symbol
		self.window3.place(x = 500, y = 540)
		print("line no. 91")
		self.T1 = tk.Label(self.root)
		self.T1.place(x = 100, y = 540)
		self.T1.config(text = "Prediction :", font = ("Arial", 30, "bold"))
		print("line no. 95")
		#self.window4 = tk.Label(self.root) # Word
		#self.window4.place(x = 230, y = 595)

		self.T2 = tk.Label(self.root)
		self.T2.place(x = 100,y = 595)
		self.T2.config(text = "Word :", font = ("Arial", 30, "bold"))
		print("line no. 102")

		self.word = ""
		self.current_symbol = "Empty"
		self.photo = "Empty"
		self.video_processing()
		print("line no. 108")


	def video_processing(self):
		ok, frame = self.vs.read()

		if ok:
			cv2image = cv2.flip(frame, 1)

			x1 = int(0.5 * frame.shape[1])
			y1 = 10
			x2 = frame.shape[1] - 10
			y2 = int(0.5 * frame.shape[1])

			cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0) ,1)
			cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)

			self.current_image = Image.fromarray(cv2image)
			imgtk = ImageTk.PhotoImage(image = self.current_image)

			self.window.imgtk = imgtk
			self.window.config(image = imgtk)
			#cv2image=cv2image.reshape(960 , 960 ,12)
			cv2image = cv2image[y1 : y2, x1 : x2]

			gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
			#print(gray.shape)
			blur = cv2.GaussianBlur(gray, (5, 5), 2)

			th3 = cv2.adaptiveThreshold(blur, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

			ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
			print(cv2image.shape)
			#print(res.shape)
			self.predict(res)  #res

			self.current_image2 = Image.fromarray(res) # res

			imgtk = ImageTk.PhotoImage(image = self.current_image2)

			self.window2.imgtk = imgtk
			self.window2.config(image = imgtk)

			self.window3.config(text = self.current_symbol,fg = "red", font = ("Arial", 30))

			#self.window4.config(text = self.word, font = ("Arial", 30))
			print("line 153")

	#		predicts = self.hs.suggest(self.word)
			
	   

		self.root.after(5, self.video_processing)

	def predict(self, test_image):

		test_image = cv2.resize(test_image, (64, 64)) #shape change

#        image = cat(3, test_image, test_image, test_image)
		image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
		result = self.loaded_model.predict(image.reshape(1, 64, 64, 3))
		print(result)
		print(result.shape)

		# result_dru = self.loaded_model_dru.predict(test_image.reshape(1 , 128 , 128 , 1))

		# result_kdi = self.loaded_model_kdi.predict(test_image.reshape(1 , 128 , 128 , 1))

		# result_smn = self.loaded_model_smn.predict(test_image.reshape(1 , 128 , 128 , 1))

		prediction = {}

		prediction['blank'] = result[0][0]

		inde = 0

		#for i in ascii_uppercase:
		for i in alpha_char:
			print(f"inde {inde} i {i}")
			prediction[i] = result[0][inde]

			inde += 1

		#LAYER 1

		prediction = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)

		self.current_symbol = prediction[0][0]
		
		if(self.current_symbol == 'blank'):

#			for i in ascii_uppercase:
			for i in alpha_char:
				self.count[i] = 0

		self.count[self.current_symbol] += 1

		if(self.count[self.current_symbol] > 25):

			#for i in ascii_uppercase:
			for i in alpha_char:
				if i == self.current_symbol:
					continue

				tmp = self.count[self.current_symbol] - self.count[i]

				if tmp < 0:
					tmp *= -1

				if tmp <= 20:
					self.count['blank'] = 0

					#for i in ascii_uppercase:
					for i in alpha_char:		
						self.count[i] = 0
					return

			self.count['blank'] = 0


			if self.current_symbol == 'blank':

				if self.blank_flag == 0:
					self.blank_flag = 1


					self.word = ""

			else:

				self.blank_flag = 0

				self.word += self.current_symbol
			
#			for i in ascii_uppercase:
			for i in alpha_char:
				self.count[i] = 0

	def action1(self):

	#	predicts = self.hs.suggest(self.word)

		if(len(predicts) > 0):

			self.word = ""

	def action2(self):

	#	predicts = self.hs.suggest(self.word)

		if(len(predicts) > 1):
			self.word = ""


	def action3(self):

	#	predicts = self.hs.suggest(self.word)

		if(len(predicts) > 2):
			self.word = ""


	def action4(self):

	#	predicts = self.hs.suggest(self.word)

		if(len(predicts) > 3):
			self.word = ""


	def action5(self):

	#	predicts = self.hs.suggest(self.word)

		if(len(predicts) > 4):
			self.word = ""
			
	def destructor(self):

		print("Closing Application...")

		self.root.destroy()
		self.vs.release()
		cv2.destroyAllWindows()
	
print("Starting Application...")

(Application()).root.mainloop()








# def recognise(cluster_model, classify_model):
#     global CAPTURE_FLAG
#     gestures = ipu.get_all_gestures()
#     cv2.imwrite("all_gestures.jpg", gestures)
#     camera = cv2.VideoCapture(0)
#     print('Now camera window will be open, then \n1) Place your hand gesture in ROI (rectangle) \n2) Press esc key to exit.')
#     count = 0
#     while(True):
#         (t,frame) = camera.read()
#         frame = cv2.flip(frame,1)
#         cv2.rectangle(frame,ipu.START, ipu.END,(0,255,0),2 )
#         cv2.imshow("All_gestures", gestures)
#         pressedKey = cv2.waitKey(1)
#         if pressedKey == 27:
#             break
#         elif pressedKey == ord('p'):
#             if(CAPTURE_FLAG):
#                 CAPTURE_FLAG = False
#             else:
#                 CAPTURE_FLAG = True
#         if(CAPTURE_FLAG):
#             print("Capturing....")
#             # Region of Interest
#             roi = frame[ ipu.START[1]+5:ipu.END[1], ipu.START[0]+5:ipu.END[0]]
#             if roi is not None:
#                 print("roi is there....")
#                 roi = cv2.resize(roi, (ipu.IMG_SIZE,ipu.IMG_SIZE))
#                 img = ipu.get_canny_edge(roi)[0]
#                 cv2.imshow("Edges ",img)
#                 print(img)
#                 sift_disc = ipu.get_SIFT_descriptors(img)
#             print(type(sift_disc))
#             if sift_disc is not None:
#                 visual_words = cluster_model.predict(sift_disc)
#                 print('visual words collected.')
#                 bovw_histogram = np.array(np.bincount(visual_words, minlength=ipu.N_CLASSES * ipu.CLUSTER_FACTOR))
#                 pred = classify_model.predict([bovw_histogram])
#                 label = class_labels[pred[0]]
#                 rectangle_bgr = (0, 0, 0)
#                 (text_width, text_height) = cv2.getTextSize('Predicted text:      ', 1, fontScale=1.5, thickness=2)[0]
#                 # set the text start position
#                 text_offset_x = 50
#                 text_offset_y = 20
#                 # make the coords of the box with a small padding of two pixels
#                 box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 40, text_offset_y + text_height +50))
#                 cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
#                 frame = cv2.putText(frame, 'Predicted text: ', (50,70), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
#                 frame = cv2.putText(frame, label, (300,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
	   
#         cv2.imshow("Video",frame)
#     camera.release()
#     cv2.destroyAllWindows()
