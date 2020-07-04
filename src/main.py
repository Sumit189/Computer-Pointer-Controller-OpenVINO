import cv2
import os
import logging
import numpy as np
import time
#Face Detection File
from face_detection import Face_DetectionModel
#Facial Landmarks Detcetion File
from facial_landmarks_detection import Facial_Landmarks_DetectionModel
#Head Pose Estimation File
from head_pose_estimation import Head_Pose_EstimationModel
#Gaze Estimation File
from gaze_estimation import Gaze_EstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    #Arguments
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help="Path of Face Detection model xml file.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help="Path of Facial Landmark Detection model xml file.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help="Path of Head Pose Estimation model xml file.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help="Path of Gaze Estimation model xml file.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path of video file or Enter cam for webcam feed")
    parser.add_argument("-v", "--visualize", required=False, nargs='+',
                        default=[],
                        help="enter the keys followed by space, ex: --v fd fld hp ge "
                             "fd: face detection model"
                             "fld: facial landmarks detection model"
                             "hp: head pose estimation model"
                             "ge: gaze estimation model " 
                             "You can enter multiple or single keys")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="CPU Extension")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Theshold for model.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on (default is CPU): "
                             "CPU, GPU, FPGA or MYRIAD is acceptable.")
    parser.add_argument("-b","--benchmark",type=str,default="false",
                        help="Pass true or false for bechmarking."
                              "If true then a file will be created containing the Model loading time, Inference Time and FPS"
                        )
    return parser

def load_models(args,logger):
    start_time2 = time.time()
    try:
        start_time = time.time()
        fdm = Face_DetectionModel(args.facedetectionmodel, args.device, args.cpu_extension)
        fdm.load_model()
        face_detect_laoding_time = (time.time() - start_time)
    except FileNotFoundError:
        logger.error("Face Detection Model doesn't exist in the path")
        exit(1)
    try:
        start_time = time.time()
        fldm = Facial_Landmarks_DetectionModel(args.faciallandmarkmodel, args.device, args.cpu_extension)
        fldm.load_model()
        facial_detect_laoding_time = (time.time() - start_time)
    except FileNotFoundError:
        logger.error("Facial Landmark Detection Model doesn't exist in the path")
        exit(1)
    try:  
        start_time = time.time()  
        hpem = Head_Pose_EstimationModel(args.headposemodel, args.device, args.cpu_extension) 
        hpem.load_model()
        head_pose_estimation_laoding_time =  (time.time() - start_time)
    except FileNotFoundError:
        logger.error("Head Pose Estimation Model doesn't exist in the path")
        exit(1) 
    try:    
        start_time = time.time()
        gem = Gaze_EstimationModel(args.gazeestimationmodel, args.device, args.cpu_extension)
        gem.load_model()
        gaze_estimation_laoding_time =  (time.time() - start_time)
    except FileNotFoundError:
        logger.error("Gaze Estimation Model doesn't exist in the path")
        exit(1)
    total_loading_time= (time.time() - start_time2)
    return fdm,fldm,gem,hpem,face_detect_laoding_time, facial_detect_laoding_time,head_pose_estimation_laoding_time,gaze_estimation_laoding_time,total_loading_time,0

def face_detect_visualize(preview_frame,face_coords):
    #boundry box for face
    boundry_box=cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
    preview_frame = boundry_box
    return preview_frame

def facial_landmarks_visualize(preview_frame,cropped_image,eye_coords,face_coords):
    #boundry boxes for eyes
    cv2.rectangle(cropped_image, (eye_coords[0][0]-5, eye_coords[0][1]-5), (eye_coords[0][2]+5, eye_coords[0][3]+5), (0,255,0), 3)
    cv2.rectangle(cropped_image, (eye_coords[1][0]-5, eye_coords[1][1]-5), (eye_coords[1][2]+5, eye_coords[1][3]+5), (0,255,0), 3)
    preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_image
    return preview_frame

def headpose_visualize(preview_frame,hp_out,face_coords):
    #Yaw, Pitch and Roll display in cv2
    text="Head Pose:-> Yaw:{:.2f} || Pitch:{:.2f} || Roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2])
    cv2.putText(preview_frame,text , (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    boundry_box=cv2.rectangle(preview_frame, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]), (255,0,0), 3)
    preview_frame = boundry_box
    return preview_frame

def gaze_visualize(preview_frame,cropped_image,x,y,left_eye,right_eye,eye_coords,face_coords):
    x, y, w = int(x*10), int(y*10), 180
    x_w_coords=(x-w,y-w)
    x_w_coords2=(x+w,y+w)
    x_w_coords3=(x-w,y+w)
    x_w_coords4=(x+w,y-w)
    left=cv2.line(left_eye.copy(),x_w_coords ,x_w_coords2, (255,0,255), 2)
    #gaze vector line for left eye
    cv2.line(le, x_w_coords3,x_w_coords4, (255,0,255), 2)
    right = cv2.line(right_eye.copy(), x_w_coords, x_w_coords2, (255,0,255), 2)
    #gaze vector line for left eye
    cv2.line(re, x_w_coords3, x_w_coords4, (255,0,255), 2)
    cropped_image[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]]=left
    cropped_image[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]]=right
    preview_frame[face_coords[1]:face_coords[3], face_coords[0]:face_coords[2]] = cropped_image
    return preview_frame

def infer_on_stream(args,logger):
    visualizers=args.visualize
    video_file=args.input
    input_feeder=None

    if video_file.lower()=="cam":
        input_feeder=InputFeeder("cam")
    else:
        try:
            input_feeder = InputFeeder("video",video_file)
        except FileNotFoundError:
            logger.error("Unable to find specified video file")
            exiit(1)
    input_feeder.load_data()
    mouse = MouseController('medium','fast')
    #load models
    fdm,fldm,gem,hpem,face_detect_laoding_time, facial_detect_laoding_time,head_pose_estimation_laoding_time,gaze_estimation_laoding_time,total_loading_time,status=load_models(args,logger) 
    if status!=0:
        #if any model is not loaded
        exit(1)     
    frame_count=0
    #start time of the inferencing
    start_inf_time = time.time()
    #iterate till the break key is pressed
    for flag, frame in input_feeder.next_batch():
        if not flag:
            break
        frame_count+=1
        if frame_count%3==0:
            cv2.imshow('video',cv2.resize(frame,(500,500)))

        key = cv2.waitKey(60)
        cropped_image, face_coords = fdm.predict(frame.copy(), args.prob_threshold)
        if type(cropped_image)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue

        hp_out = hpem.predict(cropped_image.copy()) 
        left_eye, right_eye, eye_coords = fldm.predict(cropped_image.copy())
        mouse_coord,vector = gem.predict(left_eye, right_eye, hp_out)

        if (not len(visualizers)==0):
            preview_frame = frame.copy()
            switches={"fd":0,"fld":1,"hp":2,"ge":3}
            for i in visualizers:
                val=switches.get(i)
                if val==0:
                    logger.error("Visualising: Face")
                    face_detect_visualize(preview_frame,face_coords)
                if val==1:
                    logger.error("Visualising: Facial Landmarks")
                    facial_landmarks_visualize(preview_frame,cropped_image,eye_coords,face_coords)
                if val==2:
                    logger.error("Visualising: Head Pose")
                    headpose_visualize(preview_frame,hp_out,face_coords)
                if val==3:
                    logger.error("Visualising: Gaze")
                    x=vector[0]
                    y=vector[1]
                    gaze_visualize(preview_frame,cropped_image,x,y,left_eye,right_eye,eye_coords,face_coords)
            cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))        

        if frame_count%3==0:
            mouse.move(mouse_coord[0],mouse_coord[1])   

        if key==27:
            #if benchmarking is enabled
            if (args.benchmark=="true"):
                logger.error("Face Detection Model Loading Time: {}s".format(face_detect_laoding_time))
                logger.error("Facial Landmarks Detection Model Loading Time: {}s".format(facial_detect_laoding_time))
                logger.error("Head Pose Estimation Model Loading Time: {}s".format(head_pose_estimation_laoding_time))
                logger.error("Gaze Estimation Model Loading Time: {}s".format(gaze_estimation_laoding_time))
                logger.error("Total Loading Time: {}s".format(total_loading_time))
                inference_time = round(time.time() - start_inf_time, 1)
                fps = int(frame_count) / inference_time
                logger.error("total inference time {} seconds".format(inference_time))
                logger.error("fps {} frame/second".format(fps))
                with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark.txt'), 'w') as f:
                    f.write(str("Total Inference Time: "+str(inference_time) + '\n'))
                    f.write(str("Total FPS: "+str(fps) + '\n'))
                    f.write(str("Total Model Loading Time: "+str(total_loading_time) + '\n'))
            break
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    input_feeder.close()

def main():
    # Grab command line args
    args = build_argparser().parse_args()
    #To log errors and messages
    logger=logging.getLogger()
    # Perform inference on the input stream
    infer_on_stream(args,logger)

if __name__ == '__main__':
    main() 
    