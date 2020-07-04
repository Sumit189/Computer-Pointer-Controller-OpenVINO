import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork

class Face_DetectionModel:
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

    def predict(self,image,thresh):
        self.thresh=thresh
        n,c,h,w=self.input_shape
        #coping image because we need both original and processed image
        img=image.copy()
        image=self.preprocess_input(image,n,c,h,w)
        outputs=self.exec_network.infer({self.input_blob:image})
        initial_h=img.shape[0]
        initial_w=img.shape[1]
        coordinates=self.preprocess_output(outputs,initial_w,initial_h)
        if (len(coordinates)==0):
            return 0,0
        else:
            #Getting only first face
            coordinates=coordinates[0]
            cf=img[coordinates[1]:coordinates[3], coordinates[0]:coordinates[2]]
            return cf,coordinates
    
    def preprocess_input(self, image,n,c,h,w):
        try:
            image = cv2.resize(image, (w,h))
            image = image.transpose((2, 0, 1))
            image = image.reshape(n,c,h,w)
            return image
        except Exception as e:
            print(str(e))
    
    def preprocess_output(self,output,initial_w,initial_h):
        det=[]
        for box in output[self.output_blob][0][0]:
            if box[2]>self.thresh:
                xmin = int(box[3]*initial_w)
                ymin = int(box[4]*initial_h)
                xmax = int(box[5]*initial_w)
                ymax = int(box[6]*initial_h)
                #Appending the list with coordinates of face
                det.append([xmin,ymin,xmax,ymax])
        return det