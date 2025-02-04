
from utils import get_color_structure_frames
from model import LIPINC_V2
import numpy as np
from os import listdir, path
import cv2, os, sys, argparse
import json
from datetime import datetime
import time
import os
import dlib
# from google.colab.patches import cv2_imshow


def parse_args():
  parser = argparse.ArgumentParser(description='Inference code to LIPINC models')
  parser.add_argument('--input_path', type=str, help='This path should be an external path point to an video file')
  parser.add_argument('--output_path', type=str, help='This path should be an external path point to result folder',default = "")
  parser.add_argument('--d', action="store_true", help='Use this argument when input path is a folder')
  parser.add_argument('--device', type=str, default = 'cuda', help='Set GPU or CPU as first priority')
  parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', default= "checkpoints/Best_Weights.hdf5")
  
  args = parser.parse_args()
  return args 

###################################################################################################

def get_result_description(real_p):
    if real_p <= 1 and real_p >= 0.99:
        return 'This sample is certainly real.'
    elif real_p < 0.99 and real_p >= 0.75:
        return 'This sample is likely real.'
    elif real_p < 0.75 and real_p >= 0.25:
        return 'This sample is maybe real.'
    elif real_p < 0.25 and real_p >= 0.01:
        return 'This sample is unlikely real.'
    elif real_p < 0.01 and real_p >= 0:
        return 'There is no chance that the sample is real.'
    else:
        return 'Error'

###################################################################################################
class Video_Short(Exception):
    pass

###################################################################################################
def get_result(input_path,advanced_folder_path,checkpoint_path):

  # record start time
  start = time.time()
  Error= None
  result = -1
  input_file_name = input_path.split('/')[-1]
  print("The Video being evaluated: ",input_file_name)
  try:
    advanced_video_folder_path = os.path.join(advanced_folder_path,input_file_name.split('.')[0])
    os.makedirs(advanced_video_folder_path,exist_ok= True)

    n_frames = 5 #number of local frames
    length_error,face,combined_frames,residue_frames,l_id,g_id = get_color_structure_frames(n_frames,input_path)
    if length_error:
      raise Video_Short()
    else:
      # cv2.imwrite(os.path.join(advanced_video_folder_path,'face.png'), face) 
      #saving lips image 
      for idx in range(len(combined_frames)):
            lip_image = combined_frames[idx]
            # saving the face image as a PNG file 
            cv2.imwrite(os.path.join(advanced_video_folder_path,f'lip{idx}.png'), lip_image)
            if idx< len(residue_frames):
              res_image = residue_frames[idx]  
              cv2.imwrite(os.path.join(advanced_video_folder_path,f'res{idx}.png'), res_image)

      combined_frames,residue_frames = np.reshape(combined_frames, (1,) + combined_frames.shape), np.reshape(residue_frames, (1,) + residue_frames.shape)
      print("Shape of Color Frames: {} and Structure Frames: {}".format(combined_frames.shape,residue_frames.shape))
      model = LIPINC_V2()

      model.load_weights(checkpoint_path)

      result = model.predict([combined_frames,residue_frames])
      #real probability
      result = round(float(result[0][1]),3)
      # print("Result",result)
  
  except Video_Short as e:
      Error = "Video_short"
  except Exception as e:
      Error = "no_face"
  if  Error:
    print("Error:",Error)
  # record end time
  end = time.time()
  return result,combined_frames,residue_frames,l_id,g_id,advanced_video_folder_path,round((end-start),2)

###################################################################################################
def create_zoom(frame,path,lip_bbox):
  frame_array= []
  frames_visible = 3
  ratio = min(frame.shape[0] / 1080, frame.shape[1] / 1920)
  lip_image = cv2.imread(os.path.join(path,"lip0.png"))
  lip_image = cv2.resize(lip_image, ( int(lip_image.shape[1]*ratio), int(lip_image.shape[0]*ratio)) )


  frame_h = frame.shape[0]
  frame_w = frame.shape[1]
  x,y,w,h = lip_bbox

  diff_x = frame_w//2 - x
  diff_y = frame_h//2 - y
  
  t = 1
  while(t<1.9):    

    w = int(lip_image.shape[1]*t)
    h = int(lip_image.shape[0]*t)
    new_lip_image = cv2.resize(lip_image,(w,h))
    # print(t,new_lip_image.shape)
    for i in range(frames_visible):
      background =  np.zeros([frame_h,frame_w,3],dtype=np.uint8)
      background[y:y+h,x:x+w] = new_lip_image
      # print("background",background.shape)
      frame_array.append(background)

    x = abs(x + diff_x//8)
    y = abs(y + diff_y//8)
    t+=0.1

  frame_array= np.asarray(frame_array)
  return frame_array, new_lip_image

###################################################################################################

def show_lip_extraction(background,path,lip_image,start_x,start_y):

  total_frames = 8
  images =sorted(os.listdir(path))
  
  frame_h = background.shape[0]
  frame_w = background.shape[1]
 
  ratio = background.shape[1] / (lip_image.shape[1]*(total_frames+1))

  w = int(lip_image.shape[1]*ratio)
  h = int(lip_image.shape[0]*ratio)

  # print(w,h)

  # start_x=5
  # start_y= frame.shape[0]//2

  # background =  np.zeros([frame_h,frame_w,3],dtype=np.uint8) 
  # background[:] = 255
  for image in images:
    if image[0]=="l":
      c_path = os.path.join(path,image)
      # print(c_path)
      c_image = cv2.imread(c_path)
      new_lip_image = cv2.resize(c_image,(w,h))
      background[start_y:start_y+h,start_x:start_x+w] = new_lip_image
      start_x =start_x + w + w//8
  
  start_x = 5 + w//4
  start_y= start_y + h + h//2
  for image in images:
    if image[0]=="r":
      r_path = os.path.join(path,image)
      r_image = cv2.imread(r_path)
      new_lip_image = cv2.resize(r_image,(w,h))
      background[start_y:start_y+h,start_x:start_x+w] = new_lip_image
      start_x =start_x + w + w//8
 
  return background, start_y+h


###################################################################################################


def add_text(frame,text,x,y,color):
  frame_array= []
  font = cv2.FONT_HERSHEY_SIMPLEX 
  ratio = frame.shape[1] / 1920

  # fontScale 
  fontScale = 1 * ratio
  # Line thickness of 2 px 
  thickness = int(2 * ratio)

  x = int(x * ratio)
  # y = int(y * ratio)

  frame = cv2.putText(frame,text, (x,y), font, fontScale,  
                    color, thickness)
  
  return frame,y

###################################################################################################
def create_ouput_frame(frame,advanced_video_folder_path,lip_image,video_des ):
  frame_array = []
  frames_visible =90
  frame_h = frame.shape[0]
  frame_w = frame.shape[1]
  background =  np.zeros([frame_h,frame_w,3],dtype=np.uint8) 

  Text = "Step 2: Local and Global Mouth Frame Extractor"
  background,y = add_text(background,Text,5,50,(0,0,255))
  background,y = show_lip_extraction(background,advanced_video_folder_path,lip_image,5,y+50)
  Text = "Step 3: DeepFake Detection"
  background,y = add_text(background,Text,5,y+50,(0,0,255))

  for keys in video_des:
    # print(y)
    text = keys+": "+str(video_des[keys])
    background,y = add_text(background,text,5,y+50,(255,255,255))
    

  for i in range(frames_visible):
    frame_array.append(background)
  
  
  return frame_array


###################################################################################################

def lip_extracted_slide_show(frame,advanced_video_folder_path,lip_image):
  frame_array = []
  frames_visible = 2
  frame_h = frame.shape[0]
  frame_w = frame.shape[1]
  y =  frame_h//2


  while y>=100:
    # print(y)
    background =  np.zeros([frame_h,frame_w,3],dtype=np.uint8)
    background,_ = show_lip_extraction(background,advanced_video_folder_path,lip_image,5,y)
    y=y-50
    for i in range(frames_visible):
      frame_array.append(background)
  
  
  return frame_array
###################################################################################################
def convert_2k(frame):
  
  background =  np.zeros([1080,1920,3],dtype=np.uint8)
  frame_h = frame.shape[0]
  frame_w = frame.shape[1]
  h =1080
  w = 1920
  ratio = min(w/frame_w,h/frame_h)-0.01

  frame_h = int(frame.shape[0]*ratio)
  frame_w = int(frame.shape[1]*ratio)

  frame = cv2.resize(frame,(frame_w,frame_h))
  diffh= h - frame_h
  diffw= w - frame_w

  background[diffh//2:diffh//2+frame_h,diffw//2:diffw//2+frame_w] = frame

  
  return background


###################################################################################################
def create_demo_video(input_path,color,advanced_video_folder_path,g_id,video_des):
  datFile =  "shape_predictor_68_face_landmarks.dat"
  detector_pre = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor(datFile)

  frame_array = []

  video_stream = cv2.VideoCapture(input_path)# read video
  fps = video_stream.get(cv2.CAP_PROP_FPS)# read fps

  total_frames = 0
  while 1:
    still_reading, frame = video_stream.read()
    total_frames+=1
    if not still_reading:
        video_stream.release()
        break

  # print('Total Frames in input video:',total_frames)
  
  f_id =0 
  # extract all frames
  video_stream = cv2.VideoCapture(input_path)
  while 1:
    still_reading, frame = video_stream.read()
    if not still_reading:
        video_stream.release()
        break

    frame = convert_2k(frame)
    imggr =  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    faces= detector_pre(imggr)
      
    
    try:
      top = max(0, faces[0].top())
      bottom = min(faces[0].bottom(), imggr.shape[0])
      left = max(0, faces[0].left())
      right = min(faces[0].right(), imggr.shape[1])

      landmark = predictor(imggr,faces[0])

      mypoints =[]
      for n in range(68):
        x=landmark.part(n).x
        y=landmark.part(n).y
        mypoints.append([x,y])
      points =np.array(mypoints[48:])
      bbox = cv2.boundingRect(points)
      x,y,w,h = bbox

      # image = cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2) 
      # if f_id in l_id or f_id in g_id:
      image = cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2) 

      frame_array.append(image)
      
    except Exception as e:
      frame_array.append(frame)
      continue
    
    f_id+=1
    if f_id>=total_frames-1 or  (f_id>g_id[-1] and f_id > 120):
      zoom_array, lip_image = create_zoom(frame,advanced_video_folder_path,bbox)
      break
  # print("No of frames in output video ",len(frame_array))

  frame_array= np.asarray(frame_array)
  text_set= ["Task","Input File","Analytic Name","Analysis Date"]
  for frame_id in range(len(frame_array)):
      Text = "Step 1: Local and Global Mouth Frame Matching..."
      frame_array[frame_id],y = add_text(frame_array[frame_id],Text,5,50,(0,0,255))
      for keys in video_des:
        if keys in text_set:
          text = keys+": "+str(video_des[keys])
          frame_array[frame_id],y = add_text(frame_array[frame_id],text,5,y+50,(255,255,255))

  

  collage = lip_extracted_slide_show(frame_array[0],advanced_video_folder_path,lip_image)
  lip_array = create_ouput_frame(frame_array[0],advanced_video_folder_path,lip_image,video_des)

  frame_array = np.concatenate((frame_array,zoom_array,collage,lip_array))


  return frame_array




###################################################################################################

def main():
  global args,device,output_path,input_is_folder

  args = parse_args()
  
  input_path = args.input_path
  output_path = args.output_path
  input_is_folder = args.d
  checkpoint_path = args.checkpoint_path
  device = args.device
  video_des={  "Task": "Lip-synced Deepfake Detection",
     "Input File": "",
     "Analytic Name": "LIPINC-V2",
     "Analysis Date": "",
     "1/1 [==============================] -": "N/A",
    #  "Original Result": "N/A",
    #  "Original Result Description": "The probability of input video is real",
     "Result": {
          "Real Probability": "N/A",
          "Fake Probability": "N/A"
     },
     "Result Description": "Error"
      
     }
  
  if not os.path.isfile(input_path):
    raise ValueError('--face argument must be a valid path to video/image file')

  advanced_folder_path = os.path.join(output_path,"Advanced_results") 
  os.makedirs(advanced_folder_path,exist_ok= True)
  result,combined_frames,residue_frames,l_id,g_id,advanced_video_folder_path, run_time  = get_result(input_path,advanced_folder_path,checkpoint_path)
  
  print("Result: ",result)

  input_file_name = input_path.split('/')[-1]
  video_des['Input File'] = input_file_name
  video_des['Analysis Date'] = str(datetime.now())
  if result>=0 and result<=1:
    # video_des['Original Result'] = result
    video_des['Result'] = {"Real Probability": result, "Fake Probability":  round(1-result,4)}
    video_des['Result Description'] = get_result_description(result)
    video_des["1/1 [==============================] -"] = str(run_time)+" secs"
  else:
    # video_des['Original Result'] = 'N/A'
    video_des['Result'] = {"Real Probability": 'N/A', "Fake Probability": 'N/A'}
    video_des['Result Description'] = 'Error'

  # if result > 0.5:
  #   color = (0, 255, 0)
  # else:
  #   color = (0, 0, 255)
  color = (255, 0, 0)

  
  frame_array = create_demo_video(input_path,color,advanced_video_folder_path,g_id,video_des)

  # text_array = add_text(frame_array[0],video_des)

  # print(frame_array.shape,text_array.shape)
  # frame_array = np.concatenate((frame_array,text_array))
  name = input_file_name.split('.')[0]
  demo_path =os.path.join(output_path,f"{name}_demo.mp4")
  frameSize = (frame_array.shape[2],frame_array.shape[1])
  out = cv2.VideoWriter(demo_path,cv2.VideoWriter_fourcc(*'DIVX'), 21, frameSize)
  for i in range(len(frame_array)):
    out.write(frame_array[i])
    # cv2_imshow(result[i])
  out.release()


    
    
    

if __name__ == '__main__':
    main()





