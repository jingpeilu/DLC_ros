#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import JointState
from message_filters import ApproximateTimeSynchronizer, Subscriber


import os
from skimage import io
from skimage.transform import resize
import skimage.color
from skimage.util import img_as_ubyte

from deeplabcut import DEBUG
from deeplabcut.utils import auxiliaryfunctions, conversioncode, auxfun_models, visualization
from deeplabcut.pose_estimation_tensorflow import training
from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
import tensorflow as tf

from pathlib import Path
import numpy as np
import time
import copy


def predict_single_image(image, sess, inputs, outputs, dlc_cfg):
    """
    Returns pose for one single image
    :param image:
    :return:
    """
    # The size here should be the size of the images on which your CNN was trained on
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_as_ubyte(image)
    pose = ptf_predict.getpose(image, dlc_cfg, sess, inputs, outputs)
    return pose

    
def generate_prediction(MAX_PREDICTION_STEPS = 1000):
    """
    Generator for predicting image
    MAX_PREDICTION_STEPS : Number of predictions that should be done before re-initializing 
    """

    ##################################################
    # Clone arguments from deeplabcut.evaluate_network
    ##################################################

    weights_path = "snapshot-50000" # change this
    config = "pose_cfg.yaml" # change this

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    gputouse = 0
    use_gpu = True


    # Suppress scientific notation while printing
    np.set_printoptions(suppress=True)

    ##################################################
    # SETUP everything until image prediction
    ##################################################

    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    vers = tf.__version__.split('.')
    if int(vers[0]) == 1 and int(vers[1]) > 12:
        TF = tf.compat.v1
    else:
        TF = tf

    TF.reset_default_graph()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    #tf.logging.set_verbosity(tf.logging.WARN)

    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    if gputouse is not None:  # gpu selectinon
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    path_test_config = config

    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle s and trainFraction %s does not exist.")

    dlc_cfg['init_weights'] = weights_path
    print("Running the weights: " + dlc_cfg['init_weights'])


    # Using GPU for prediction
    # Specifying state of model (snapshot / training state)
    if use_gpu:
        sess, inputs, outputs = ptf_predict.setup_GPUpose_prediction(dlc_cfg)
        pose_tensor = ptf_predict.extract_GPUprediction(outputs, dlc_cfg)
    else:
        sess, inputs, outputs = ptf_predict.setup_pose_prediction(dlc_cfg)

    print("Analyzing test image ...")
    imagename = "img034.png"
    image = io.imread(imagename, plugin='matplotlib')

    count = 0
    start_time = time.time()
    while count < MAX_PREDICTION_STEPS:

        ##################################################
        # Predict for test image once, and wait for future images to arrive
        ##################################################
        
        print("Calling predict_single_image: " + str(count))
        if use_gpu:       
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = img_as_ubyte(image)
            pose = sess.run(pose_tensor, feed_dict={inputs: np.expand_dims(image, axis=0).astype(float)})
            pose[:, [0,1,2]] = pose[:, [1,0,2]]
        else:
            pose = predict_single_image(image, sess, inputs, outputs, dlc_cfg)
            
            


        #print(pose)
        ##################################################
        # Yield prediction to caller
        ##################################################
        
        image = (yield pose) # Receive image here ( Refer https://stackabuse.com/python-generators/ for sending/receiving in generators)
        
        #step_time = time.time()
        #start_time = step_time
        count += 1
        print(count)

        if count == MAX_PREDICTION_STEPS:
            print(f"Restart prediction system, Steps have exceeded {MAX_PREDICTION_STEPS}")

    sess.close()  # closes the current tf session
    TF.reset_default_graph()



class image_converter:

  def __init__(self, generator_1):
    self.image_pub = rospy.Publisher("/dlc_prediction_topic", Image,queue_size=10) # change this
    self.bridge = CvBridge()
    #self.image_sub = rospy.Subscriber("/stereo/slave/left/image", Image, self.callback)
    self.image_sub_left = rospy.Subscriber("/stereo/left/rectified_downscaled_image", Image, self.callback_left) # change this
    #self.ats = ApproximateTimeSynchronizer([self.image_sub_left, self.image_sub_right], queue_size=5, slop=0.1)
    #self.ats.registerCallback(self.callback)
    self.generator_left = generator_1
    self.pose_pub_left = rospy.Publisher("/dlc_pose_array_left", JointState,queue_size=10) # change this


  def callback_left(self,data_l):
    try:
      self.cv_image_l = self.bridge.imgmsg_to_cv2(data_l, "bgr8")
      #self.cv_image_r = self.bridge.imgmsg_to_cv2(data_r, "bgr8")

      #temp_image = copy.deepcopy(self.cv_image)
      #self.cv_image_l, _ = rectify_undistort(self.cv_image_l, self.cv_image_l)
      #temp_image_l = cv2.resize(self.cv_image_l, (640,480))
      #temp_image_r = cv2.resize(self.cv_image_r, REQUIRED_DIM)

    except CvBridgeError as e:
      print(e)

    # Generator approach
    try:
      results_l = self.generator_left.send(self.cv_image_l)
      points_predicted_l = results_l[:,:2]
      scores_l = results_l[:,2]
      #results_r = self.generator.send(temp_image_r)
      #points_predicted_r = results_r[:,:2]
      #scores_r = results_r[:,2]

      #print(results)
      #pass     
    except ValueError as e:
        if str(e) == 'generator already executing':
           print("Prediction ongoing, returning previous image")
           return 

    #points_predicted_l = self.modify_points_predicted(points_predicted_l)
    #points_predicted_r = self.modify_points_predicted(points_predicted_r)

    
    # convert prediction to ros pose array
    ps = JointState()
    ps.header.stamp = data_l.header.stamp
    # right_shaft_up, right_shaft_tip, right_logo_body_up, right_logo_body_tail_tip, right_logo_body_head_tip, right_arm_right_jaw_edge, right_arm_left_jaw_center
    ps.name = ['PSM1-yaw_1_front', 'PSM1-yaw_2_front', 'PSM1-pitch_1_front', 'PSM1-pitch_2_front', 'PSM1-pitch_3_front', 'PSM1-roll_1_front','PSM1-roll_2_front','PSM1-yaw_1_back', 'PSM1-yaw_2_back', 'PSM1-pitch_1_back', 'PSM1-pitch_2_back', 'PSM1-pitch_3_back', 'PSM1-roll_1_back','PSM1-roll_2_back'] # change this
    ps.position = list(points_predicted_l[:,0])# + list(points_predicted_r[:,0]) # x coordinates
    ps.velocity = list(points_predicted_l[:,1])# + list(points_predicted_r[:,1]) # y coordinates
    ps.effort = scores_l# + scores_r
    
    temp_pub_img = self.overwrite_image(self.cv_image_l, points_predicted_l ,scores_l) 
    
    # PUBLISH 
    self.pose_pub_left.publish(ps)
    try:
      image_message = self.bridge.cv2_to_imgmsg(temp_pub_img, "bgr8")
      image_message.header.stamp = data_l.header.stamp
      self.image_pub.publish(image_message)
    except CvBridgeError as e:
      print(e)

  
  def modify_points_predicted(self,points_predicted):
    """Modify each point predicted to convert it back from the predicted size to the original
       size of the image that came from ROS node"""
    
    MODIFYING_SCALE = 2
    points_predicted[:,0] *= MODIFYING_SCALE
    points_predicted[:,1] *= MODIFYING_SCALE  # After cropping the image was halved, so doubling it back up!
    return points_predicted

  def overwrite_image(self,image, points_predicted,scores):
    """For each point in points_predicted make four corners and overwrite those 4 points in the cv_image with blue markers"""

    # TODO: Separate this to another function
    height, width = image.shape[:2]

    #Clipping points so that they don't fall outside the image size
    #points_predicted[:,0] = points_predicted[:,0].clip(0, height-1)
    #points_predicted[:,1] = points_predicted[:,1].clip(0, width-1)

    points_predicted = points_predicted.astype(int)

    # Printing as a circle
    for i in range(len(points_predicted)):
        #print(points)
        points = points_predicted[i]
        image = cv2.circle(image,tuple(points), 10, (0,0,255), -1)
        image = cv2.putText(image, str(round(scores[i],3)), tuple(points), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 1, cv2.LINE_AA)  
    return image


def main(args):
  MAX_PREDICTION_STEPS = int(1e5)

  # Initialize and kickstart generator to test the first saved image
  generator_1 = generate_prediction(MAX_PREDICTION_STEPS)
  points_predicted = generator_1.send(None)
  print(f"First prediction: {points_predicted}")


  ic = image_converter(generator_1)
  rospy.init_node('image_converter', anonymous=True)

  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)


# conda activate dlc-ubuntu-GPU
# source ~/autosurg_ws/devel/setup.bash
# export ROS_MASTER_URI=http://192.168.0.104:11311
# export ROS_IP=192.168.0.102

