# Blip import
import torch
import torchvision
import os
import random 
import numpy as np
import argparse
import decord

from einops import rearrange
from torchvision import transforms
from tqdm import tqdm
from PIL import Image as PIL_Image
from decord import cpu
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from cv_bridge import CvBridge,CvBridgeError

decord.bridge.set_bridge('torch')

# ros2 import
import rclpy

from rclpy.node import Node
from rclpy import qos
from std_msgs.msg import String
from sensor_msgs.msg import Image as Sensor_Image

class VideoBlipNode(Node):
    
    def __init__(self):
        super().__init__('video_blip_node')
        # rosparam initialization
        self.declare_parameter('image_topic_name', '/image_raw')
        self.declare_parameter('output_text_topic', '/blip/data')
        self.declare_parameter('model_name', 'kpyu/video-blip-opt-2.7b-ego4d')
        self.declare_parameter('question', '')

        # read params
        self.image_topic = self.get_parameter('image_topic_name').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_text_topic').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.prompt = self.get_parameter('question').get_parameter_value().string_value


        # pub sub
        self.image_subscription = self.create_subscription(Sensor_Image, self.image_topic, self.image_callback, qos.qos_profile_sensor_data)
        self.blip_publisher = self.create_publisher(String, self.output_topic, 10)

        #other params
        self.runnimg = False
        self.processor = None
        self.blip_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bridge = CvBridge()

    def load_model(self):
        '''
        Load BLIP2 model
        '''
        self.get_logger().info('Loading model')

        processor = Blip2Processor.from_pretrained(self.model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        model.to(self.device)

        self.processor = processor
        self.blip_model = model
        self.get_logger().info('Loading end')

    def sensor_msg_convert_PIL(self, input_image: Sensor_Image):
        '''
        convert
        sensor Image -> PIL Image
        '''
        # self.get_logger().info('convert')
        self.runnimg = True
        try:
            cv_image = self.bridge.imgmsg_to_cv2(input_image, "bgr8")
        except CvBridgeError as e:
            print(e)

        pil_image = cv_image[:, :, ::-1]

        return pil_image
    
    def process_blip(self, image: PIL_Image):
        '''
        process blip and generate text
        '''
        self.get_logger().info('process')
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.blip_model.generate(
                **inputs
            )
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True)[0].strip()
        
        return generated_text
    
    def image_callback(self, msg):
        self.get_logger().info('Subscription image')
        if not self.runnimg:
            pil_image = self.sensor_msg_convert_PIL(msg)
            get_text = self.process_blip(pil_image)
            pub_msg = String()
            pub_msg.data = get_text
            self.blip_publisher.publish(pub_msg)
            self.runnimg = False



def main():
    rclpy.init()
    node = VideoBlipNode()
    node.load_model()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown


if __name__ == '__main__':
    main()
