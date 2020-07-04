import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork

class Head_Pose_EstimationModel:
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
            self.input_blob=next(iter(self.network.inputs))
            self.input_shape=self.network.inputs[self.input_blob].shape
            self.output_blob=next(iter(self.network.outputs))
            self.output_shape=self.network.outputs[self.output_blob].shape
    
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

    def predict(self,image):
        n,c,h,w=self.input_shape
        #coping image because we need both original and processed image
        img=image.copy()
        image=self.preprocess_input(image,n,c,h,w)
        outputs=self.exec_network.infer({self.input_blob:image})
        #getting the headpose
        result=self.preprocess_output(outputs)
        return result
    def preprocess_input(self, image,n,c,h,w):
        try:
            image = cv2.resize(image, (w,h))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n,c,h,w)
            return image
        except Exception as e:
            print(str(e))
    
    def preprocess_output(self,detection_result):
        #conversion to list was important because only integer scalar arrays can be converted to a scalar index
        x=[0]*3
        result=[]
        x[0]=detection_result['angle_y_fc'].tolist()[0][0]
        x[1]=detection_result['angle_p_fc'].tolist()[0][0]
        x[2]=detection_result['angle_r_fc'].tolist()[0][0]
        for i in range(0,3):
            result.append(x[i])      
        return result
    

