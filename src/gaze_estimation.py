import cv2
import math
import numpy as np
from openvino.inference_engine import IECore, IENetwork

class Gaze_EstimationModel:
    def __init__(self, model, device='CPU', extensions=None):
        self.model_name=model
        self.device=device
        self.extensions=extensions
        self.plugin=None
        self.net=None
        self.thresh=None
        self.exec_network=None
        self.input_blob=None
        self.output_blob=None
        self.input_shape=None
        self.output_shape=None

    def load_model(self):
        self.model_structure=self.model_name
        self.model_weights=self.model_name.split('.')[0]+'.bin'
        self.plugin=IECore()

        #Loading the network
        self.network=self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        #adding plugins
        if self.extensions and self.device=="CPU":
            self.plugin.add_extension(self.extensions,self.device)
        #checking for unsupported layers
        status=self.checkLayers()
        if status!=0:
            exit(1)
        else:
            self.exec_network=self.plugin.load_network(network=self.network,device_name=self.device,num_requests=1)
            input_keys=self.network.inputs.keys()
            output_keys=self.network.outputs.keys()
            self.input_blob=[x for x in input_keys]
            self.input_shape=self.network.inputs[self.input_blob[1]].shape
            self.output_blob=[x for x in output_keys]
            # self.output_shape=self.network.outputs[self.output_blob].shape
    
    def checkLayers(self):
        keys=self.network.layers.keys()
        supported_layers=self.plugin.query_network(network=self.network,device_name=self.device)
        unsupported_layers=[l for l in keys if l not in supported_layers]
        if len(unsupported_layers)!=0:
            #unsupporetd layers found
            print("Unsupported Layers: {}".format(unsupported_layers))
            return 1
        else:
            return 0

    def predict(self,l_eye_image,r_eye_image,head_pose):
        n,c,h,w=self.input_shape
        #coping image because we need both original and processed image
        limg=l_eye_image.copy()
        rimg=r_eye_image.copy()
        l_eye_image=self.preprocess_input(l_eye_image,n,c,h,w)
        r_eye_image=self.preprocess_input(r_eye_image,n,c,h,w)
        #getting the coordinates for mouse
        outputs=self.exec_network.infer({'head_pose_angles':head_pose,'left_eye_image':l_eye_image,'right_eye_image':r_eye_image})
        sin,cos,vector=self.preprocess_output(outputs,head_pose)
        #x_coordinates
        x_axis=vector[0]*cos+vector[1]*sin
        #y_coordinates
        y_axis=-vector[0]*sin+vector[1]*cos
        mouse_coords=(x_axis,y_axis)
        return mouse_coords,vector

    def preprocess_input(self, image,n,c,h,w):
        try:
            image = cv2.resize(image, (w,h))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n,c,h,w)
            return image
        except Exception as e:
            print(str(e))
    
    def preprocess_output(self,detection_result,head_pose):
        #conversion to list was important because only integer scalar arrays can be converted to a scalar index    
        vector=detection_result[self.output_blob[0]].tolist()[0]
        roll=head_pose[2]
        val=roll*3.14/(360.0/2.0)
        cosVal=math.cos(val)
        sinVal=math.sin(val)
        return sinVal,cosVal,vector
    

