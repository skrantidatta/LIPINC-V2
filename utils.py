
# from google.colab.patches import cv2_imshow # comment this if not using colab
import os
import dlib
from glob import glob
import shutil
import imutils
import cv2
import numpy as np
from os import listdir
import math
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
from imutils.face_utils import FACIAL_LANDMARKS_IDXS
from imutils.face_utils import shape_to_np
from sklearn.metrics import roc_curve, auc
import sys



datFile =  "shape_predictor_68_face_landmarks.dat"
detector_pre = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(datFile)
# n_frames = 5 #number of local frames



"""##Creating Local frames and Global frames"""

class FaceAligner:
    def __init__(self, predictor, desiredLeftEye=(0.40, 0.40),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = shape_to_np(shape)
        # extract the left and right eye (x, y)-coordinates
        (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
        leftEyePts = shape[lStart:lEnd]
        rightEyePts = shape[rStart:rEnd]
        # compute the center of mass for each eye
        leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
        rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist


        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
            int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output



###################################################################################################

def create_face_array(path,fa):

  face_array = []

  video_stream = cv2.VideoCapture(path)# read video
  fps = video_stream.get(cv2.CAP_PROP_FPS)# read fps 

  while 1:
      still_reading, frame = video_stream.read()
      if not still_reading:
          video_stream.release()
          break
      
      try:
        image = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector_pre(gray)
        faceAligned = fa.align(image, gray, rects[0])
      except Exception as e:
        continue

      # frame = cv2.resize(faceAligned, (224,224))
      frame = faceAligned
      face_array.append(frame)

  return  np.asarray(face_array)

###################################################################################################

def find_global_frames(face_array,local_face_id,lh,lw,lch,rch,predictor):

    simPoseVideos = []
    g_id = []
    face_id =-1
    while face_id < (len(face_array))-1:
      face_id+=1
      if face_id == local_face_id:
        continue

      frame = face_array[face_id]
      imggr =  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

      faces= detector_pre(imggr)

      top = max(0, faces[0].top())
      bottom = min(faces[0].bottom(), imggr.shape[0])
      left = max(0, faces[0].left())
      right = min(faces[0].right(), imggr.shape[1])

      # try:
      landmark = predictor(imggr,faces[0])

      mypoints =[]
      for n in range(68):
        x=landmark.part(n).x
        y=landmark.part(n).y
        mypoints.append([x,y])
      points =np.array(mypoints[48:])
      bbox = cv2.boundingRect(points)
      x,y,w,h = bbox
      lipU_x,lipU_y = landmark.part(62).x,landmark.part(62).y
      lipL_x,lipL_y = landmark.part(66).x,landmark.part(66).y

      lipLeft_x,lipLeft_y = landmark.part(60).x,landmark.part(60).y
      lipRight_x,lipRight_y = landmark.part(64).x,landmark.part(64).y

      Outerlip_Left_x,Outerlip_Left_y = landmark.part(48).x,landmark.part(48).y
      CheekLeft_x, CheekLeft_y = landmark.part(4).x,landmark.part(4).y


      Outerlip_Right_x,Outerlip_Right_y = landmark.part(54).x,landmark.part(54).y
      CheekRight_x, CheekRight_y = landmark.part(12).x,landmark.part(12).y

      total_width = right -left
      total_height = bottom-top

      #calculating the distance and ratios between lower and upper lips
      lheight = int(math.dist([lipU_x,lipU_y],[lipL_x,lipL_y])/total_height*100)
      lwidth = int(math.dist([lipLeft_x,lipLeft_y],[lipRight_x,lipRight_y])/total_width*100)
      lcheek = int(math.dist([Outerlip_Left_x,Outerlip_Left_y],[CheekLeft_x,CheekLeft_y])/total_width*100)
      rcheek= int(math.dist([Outerlip_Right_x,Outerlip_Right_y],[CheekRight_x,CheekRight_y])/total_width*100)

      # print(int(lh*100),lheight )
      # print(int(lw*100),lwidth )
      # print(int(lch*100),lcheek )
      # print(int(rch*100),rcheek )
      ran = 3

      if int(lh*100) in range(lheight-ran,lheight+ran) and int(lw*100) in range(lwidth-ran,lwidth+ran) and int(lch*100) in range(lcheek-ran,lcheek+ran) and int(rch*100) in range(rcheek-ran,rcheek+ran):
        lipcrop = frame[y:y+h,x:x+w]
        lipcrop = cv2.resize(lipcrop, (144,64))
        simPoseVideos.append(lipcrop)
        g_id.append(face_id)
        face_id+=3


        if len(simPoseVideos) == 3:
          return simPoseVideos,g_id

      # except Exception as e:
      #   continue
    return 0,0

###################################################################################################

def find_LGframes(n_frames,face_array,predictor):

  #saving the local frames and global
  Local_frames = 0
  Global_frames = 0

  adj_f = []
  adj_f_id =[]
  No_face_count = 0

  # extract all frames
  for face_id in range(len(face_array)):


      frame = face_array[face_id]
      imggr =  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

      faces= detector_pre(imggr)
      success = 1

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

        lipU_x,lipU_y = landmark.part(62).x,landmark.part(62).y
        lipL_x,lipL_y = landmark.part(66).x,landmark.part(66).y

        lipLeft_x,lipLeft_y = landmark.part(60).x,landmark.part(60).y
        lipRight_x,lipRight_y = landmark.part(64).x,landmark.part(64).y

        Outerlip_Left_x,Outerlip_Left_y = landmark.part(48).x,landmark.part(48).y
        CheekLeft_x, CheekLeft_y = landmark.part(4).x,landmark.part(4).y


        Outerlip_Right_x,Outerlip_Right_y = landmark.part(54).x,landmark.part(54).y
        CheekRight_x, CheekRight_y = landmark.part(12).x,landmark.part(12).y

        total_width = right -left
        total_height = bottom-top


        #calculating the distance between lower and upper lips
        lheight = math.dist([lipU_x,lipU_y],[lipL_x,lipL_y])/total_height
        lwidth = math.dist([lipLeft_x,lipLeft_y],[lipRight_x,lipRight_y])/total_width
        lcheek = math.dist([Outerlip_Left_x,Outerlip_Left_y],[CheekLeft_x,CheekLeft_y])/total_width
        rcheek= math.dist([Outerlip_Right_x,Outerlip_Right_y],[CheekRight_x,CheekRight_y])/total_width


        height_ratio = lheight*100
        if height_ratio < 1:
          adj_f = []
          # success+=1 
          # if success>2:
          #   adj_f = []
          continue
        else:
          lipcrop = frame[y:y+h,x:x+w]
          lipcrop = cv2.resize(lipcrop, (144,64))
          adj_f.append(lipcrop)
          adj_f_id.append(face_id)
        
          # cv2_imshow(lipcrop)
          # print(len(adj_f))
          if len(adj_f)==n_frames:
            print("Local Frames = 5")
            Global_frames,g_id = find_global_frames(face_array,face_id,lheight,lwidth,lcheek,rcheek,predictor)
            if Global_frames == 0:
              adj_f = []
              adj_f_id = []
              continue
            Local_frames= adj_f
            break

      except Exception as e:
        No_face_count+=1
        
  print("Face/lips not Detected in {} out of {} frames".format(No_face_count,len(face_array)))
  
  Global_frames = np.asarray(Global_frames)
  Local_frames = np.asarray(Local_frames)

  # print(Global_frames.shape,Local_frames.shape)

  return Local_frames,Global_frames,adj_f_id,g_id

###################################################################################################

def create_residue(frames):
  residue = []
  for index in range(1,len(frames)):
    residue_frame = abs(frames[index]-frames[index-1])
    residue.append(residue_frame)
  residue = np.asarray(residue)
  return residue

###################################################################################################

def get_color_structure_frames(n_frames,path):
  """
  input : 
    1. n_frames: number of local frames
    2. path: input video path
  
  returns: 
    1. length error: boolean value. True if the input video is less than 5 secs
    2. face_array[0]: face of the deepfaked person
    3. combined_frames: Color lip-crops used for testing 
    4. residue_frames: structure lip-crops used for testing
    5. l_id: frame id of local frames
    6. g_id: frame id of global frames

  """

  length_error = False
  fa = FaceAligner(predictor, desiredFaceWidth=256)
  face_array = create_face_array(path,fa)
  print("Number of frames with faces",len(face_array))
  if len(face_array)<31:
    length_error = True
    return length_error,[],[],[],[],[]
  
  # cv2.imwrite(os.path.join('/content/','face.png'), face_array[0]) 
  Local_frames , Global_frames, l_id,g_id = find_LGframes(n_frames,face_array,predictor)  #find local and global frames

  combined_frames = np.concatenate((Local_frames,Global_frames))
  residue_frames = create_residue(combined_frames)
  

  return length_error,face_array[0],combined_frames,residue_frames,l_id,g_id

# combined_frames,residue_frames = get_color_structure_frames(n_frames,path)