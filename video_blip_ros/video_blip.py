# Blip import
import torch
import torch.nn as nn
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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BatchEncoding,
    Blip2Config,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    Blip2QFormerModel,
    Blip2VisionModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from cv_bridge import CvBridge,CvBridgeError

decord.bridge.set_bridge('torch')

# ros2 import
import rclpy

from rclpy.node import Node
from rclpy import qos
from std_msgs.msg import String
from sensor_msgs.msg import Image as Sensor_Image

def process(
    processor: Blip2Processor,
    video: torch.Tensor | None = None,
    text: str | list[str] | None = None,
) -> BatchEncoding:
    """Process videos and texts for VideoBLIP.

    :param images: a tensor of shape (batch, channel, time, height, width) or
        (channel, time, height, width)
    """
    if video is not None:
        if video.dim() == 4:
            video = video.unsqueeze(0)
        batch, channel, time, _, _ = video.size()
        video = video.permute(0, 2, 1, 3, 4).flatten(end_dim=1)
    print(str(video.size()))
    inputs = processor(images=video, text=text, return_tensors="pt")
    if video is not None:
        _, _, height, weight = inputs.pixel_values.size()
        inputs["pixel_values"] = inputs.pixel_values.view(
            batch, time, channel, height, weight
        ).permute(0, 2, 1, 3, 4)
    return inputs


class VideoBlipVisionModel(Blip2VisionModel):
    """A simple, augmented version of Blip2VisionModel to handle videos."""

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        """Flatten `pixel_values` along the batch and time dimension, pass it
        through the original vision model, then unflatten it back.

        :param pixel_values: a tensor of shape (batch, channel, time, height, width)

        :returns:
            last_hidden_state: a tensor of shape (batch, time * seq_len, hidden_size)
            pooler_output: a tensor of shape (batch, time, hidden_size)
            hidden_states:
                a tuple of tensors of shape (batch, time * seq_len, hidden_size),
                one for the output of the embeddings + one for each layer
            attentions:
                a tuple of tensors of shape (batch, time, num_heads, seq_len, seq_len),
                one for each layer
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        batch, _, time, _, _ = pixel_values.size()

        # flatten along the batch and time dimension to create a tensor of shape
        # (batch * time, channel, height, width)
        flat_pixel_values = pixel_values.permute(0, 2, 1, 3, 4).flatten(end_dim=1)

        vision_outputs: BaseModelOutputWithPooling = super().forward(
            pixel_values=flat_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # now restore the original dimensions
        # vision_outputs.last_hidden_state is of shape
        # (batch * time, seq_len, hidden_size)
        seq_len = vision_outputs.last_hidden_state.size(1)
        last_hidden_state = vision_outputs.last_hidden_state.view(
            batch, time * seq_len, -1
        )
        # vision_outputs.pooler_output is of shape
        # (batch * time, hidden_size)
        pooler_output = vision_outputs.pooler_output.view(batch, time, -1)
        # hidden_states is a tuple of tensors of shape
        # (batch * time, seq_len, hidden_size)
        hidden_states = (
            tuple(
                hidden.view(batch, time * seq_len, -1)
                for hidden in vision_outputs.hidden_states
            )
            if vision_outputs.hidden_states is not None
            else None
        )
        # attentions is a tuple of tensors of shape
        # (batch * time, num_heads, seq_len, seq_len)
        attentions = (
            tuple(
                hidden.view(batch, time, -1, seq_len, seq_len)
                for hidden in vision_outputs.attentions
            )
            if vision_outputs.attentions is not None
            else None
        )
        if return_dict:
            return BaseModelOutputWithPooling(
                last_hidden_state=last_hidden_state,
                pooler_output=pooler_output,
                hidden_states=hidden_states,
                attentions=attentions,
            )
        return (last_hidden_state, pooler_output, hidden_states, attentions)


class VideoBlipForConditionalGeneration(Blip2ForConditionalGeneration):
    def __init__(self, config: Blip2Config) -> None:
        # HACK: we call the grandparent super().__init__() to bypass
        # Blip2ForConditionalGeneration.__init__() so we can replace
        # self.vision_model
        super(Blip2ForConditionalGeneration, self).__init__(config)

        self.vision_model = VideoBlipVisionModel(config.vision_config)

        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size)
        )
        self.qformer = Blip2QFormerModel(config.qformer_config)

        self.language_projection = nn.Linear(
            config.qformer_config.hidden_size, config.text_config.hidden_size
        )
        if config.use_decoder_only_language_model:
            language_model = AutoModelForCausalLM.from_config(config.text_config)
        else:
            language_model = AutoModelForSeq2SeqLM.from_config(config.text_config)
        self.language_model = language_model

        # Initialize weights and apply final processing
        self.post_init()

class VideoBlipNode(Node):
    
    def __init__(self):
        super().__init__('video_blip_node')
        # rosparam initialization
        self.declare_parameter('image_topic_name', '/image_raw')
        self.declare_parameter('output_text_topic', '/blip/data')
        self.declare_parameter('model_name', 'kpyu/video-blip-opt-2.7b-ego4d')
        self.declare_parameter('question', '')
        self.declare_parameter('sensory_memory',15)
        self.declare_parameter('camera_height', 1080)
        self.declare_parameter('camera_width', 1920)
        self.declare_parameter('max_count', 30)

        # read params
        self.image_topic = self.get_parameter('image_topic_name').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_text_topic').get_parameter_value().string_value
        self.model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.prompt = self.get_parameter('question').get_parameter_value().string_value
        self.camera_fps = self.get_parameter('sensory_memory').get_parameter_value().integer_value
        self.camera_height = self.get_parameter('camera_height').get_parameter_value().integer_value
        self.camera_width = self.get_parameter('camera_width').get_parameter_value().integer_value
        self.count = self.get_parameter('max_count').get_parameter_value().integer_value
        


        # pub sub
        self.image_subscription = self.create_subscription(Sensor_Image, self.image_topic, self.image_callback, qos.qos_profile_sensor_data)
        self.blip_publisher = self.create_publisher(String, self.output_topic, 10)

        #other params
        self.runnimg = False
        self.processor = None
        self.blip_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bridge = CvBridge()
        self.camera_color_buffer_size = 3
        self.input_images = torch.zeros(self.camera_fps, 3, self.camera_height, self.camera_width)
        self.t = 0

    def load_model(self):
        '''
        Load BLIP2 model
        '''
        self.get_logger().info('Loading model')

        processor = Blip2Processor.from_pretrained(self.model_name)
        model = VideoBlipForConditionalGeneration.from_pretrained(
            self.model_name
        ).to(self.device)

        self.processor = processor
        self.blip_model = model
        self.get_logger().info('Loading end')

    def sensor_msg_convert_PIL(self, input_image: Sensor_Image):
        '''
        convert
        sensor Image -> PIL Image
        '''
        # self.get_logger().info('convert')
        try:
            cv_image = self.bridge.imgmsg_to_cv2(input_image, "bgr8")
        except CvBridgeError as e:
            print(e)

        # pil_image = torch.from_numpy(cv_image[:, :, ::-1].copy()).unsqueeze(0)
        # self.get_logger().info(str(pil_image.size()))
        self.input_images[0:-1] = self.input_images[1:].clone()
        self.input_images[-1] = torch.from_numpy(cv_image[:, :, ::-1].copy()).permute(2,0,1)
        # self.get_logger().info(str(self.input_images.size()))
        pil_image = self.input_images.clone()
        return pil_image.permute(1,0,2,3)
    
    def process_blip(self, image: PIL_Image):
        '''
        process blip and generate text
        '''
        self.get_logger().info('process')
        inputs = process(self.processor, video=image, text=self.prompt).to(self.blip_model.device)
        generated_ids = self.blip_model.generate(
                **inputs
            )
        generated_text = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True)[0].strip()
        
        return generated_text
    
    def image_callback(self, msg):
        self.get_logger().info('Subscription image')
        pil_image = self.sensor_msg_convert_PIL(msg)
        if self.t == self.count:
            if not self.runnimg:
                self.runnimg = True
                get_text = self.process_blip(pil_image)
                pub_msg = String()
                pub_msg.data = get_text
                self.blip_publisher.publish(pub_msg)
                self.runnimg = False
            self.t=0
                
        elif self.t > self.count:
            self.get_logger().info('Wait process')
        else:
            self.t= self.t+1



def main():
    rclpy.init()
    node = VideoBlipNode()
    node.load_model()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown


if __name__ == '__main__':
    main()
