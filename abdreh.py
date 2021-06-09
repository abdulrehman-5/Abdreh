# Importing required libraries, obviously
from pathlib import Path
import configparser
import tensorflow as tf
import threading
import video_utils
import sys

import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

def model_load_into_memory(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(path_to_ckpt), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return detection_graph

def run_inference_for_single_image(image, sess, graph, class_id=None):
    """Feed forward an image into the object detection model.
    
    Args:
        image (ndarray): Input image in numpy format (OpenCV format).
        sess: TF session.
        graph: Object detection model loaded before.
        class_id (list): Optional. Id's of the classes you want to detect. 
            Refer to mscoco_label_map.pbtxt' to find out more.
        
    Returns:
        output_dict (dict): Contains the info related to the detections.
            num_detections (int): Fixed to 100 for this net.
            detection_boxes (2D-ndarray): 100 arrays containing the detecion
                bounding boxes like [ymin, xmin, ymax, xmax] from 0 to 1.
            detection_scores (ndarray): Prediction scores associated with
                every detection.
            detection_classes (ndarray): Class' ID associated with
                every detection.
        
    """
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0].astype(np.float32)

    return output_dict

def discriminate_class(output_dict, classes_to_detect, category_index):
    """Keeps the classes of interest of the frame and ignores the others
    
    Args:
        output_dict (dict): Output of the model once an image is processed.
        classes_to_detect (list): Names of the classes to be detected.
        category_index (dict): Contains X dicts corresponding to each one
            of the classes where the model's been trained on. 
        
    Returns:
        output_dict (dict): Modified dictionary which just delivers the
            specified class detections.
            
    """
    for i in range(output_dict['detection_classes'].size):
        class_detected = category_index[output_dict['detection_classes'][i]]['name']
        if output_dict['detection_scores'][i]>=0.5 and class_detected not in classes_to_detect:
            # The detection is from the desired class and with enough confidence
            # Decrease the detection confidence to 0 to avoid displaying it
            output_dict['detection_scores'][i] = 0.0
       
    return output_dict
            
def visualize_results(image, output_dict, category_index):
    """Returns the resulting image after being passed to the model.
    
    Args:
        image (ndarray): Original image given to the model.
        output_dict (dict): Dictionary with all the information provided 
            by the model.
        category_index (dict): Contains X dicts corresponding to each one
            of the classes where the model's been trained on.
    
    Returns:
        image (ndarray): Visualization of the results form above.
        
    """
    vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
    
    return image


def main():
    # Initialization
    ## Load the configuration variables from 'config.ini'
    config = configparser.ConfigParser()
    config.read('config.ini')
    ## Loading label map
    num_classes = config.getint('net', 'num_classes')
    path_to_labels = config['net']['path_to_labels']
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
        max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

def model_load_into_memory(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(str(path_to_ckpt), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            
    return detection_graph

def run_inference_for_single_image(image, sess, graph, class_id=None):
    """Feed forward an image into the object detection model.
    
    Args:
        image (ndarray): Input image in numpy format (OpenCV format).
        sess: TF session.
        graph: Object detection model loaded before.
        class_id (list): Optional. Id's of the classes you want to detect. 
            Refer to mscoco_label_map.pbtxt' to find out more.
        
    Returns:
        output_dict (dict): Contains the info related to the detections.
            num_detections (int): Fixed to 100 for this net.
            detection_boxes (2D-ndarray): 100 arrays containing the detecion
                bounding boxes like [ymin, xmin, ymax, xmax] from 0 to 1.
            detection_scores (ndarray): Prediction scores associated with
                every detection.
            detection_classes (ndarray): Class' ID associated with
                every detection.
        
    """
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 
                'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
        
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(image, 0)})
    
    # All outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0].astype(np.float32)

    return output_dict

def discriminate_class(output_dict, classes_to_detect, category_index):
    """Keeps the classes of interest of the frame and ignores the others
    
    Args:
        output_dict (dict): Output of the model once an image is processed.
        classes_to_detect (list): Names of the classes to be detected.
        category_index (dict): Contains X dicts corresponding to each one
            of the classes where the model's been trained on. 
        
    Returns:
        output_dict (dict): Modified dictionary which just delivers the
            specified class detections.
            
    """
    for i in range(output_dict['detection_classes'].size):
        class_detected = category_index[output_dict['detection_classes'][i]]['name']
        if output_dict['detection_scores'][i]>=0.5 and class_detected not in classes_to_detect:
            # The detection is from the desired class and with enough confidence
            # Decrease the detection confidence to 0 to avoid displaying it
            output_dict['detection_scores'][i] = 0.0
       
    return output_dict
            
def visualize_results(image, output_dict, category_index):
    """Returns the resulting image after being passed to the model.
    
    Args:
        image (ndarray): Original image given to the model.
        output_dict (dict): Dictionary with all the information provided 
            by the model.
        category_index (dict): Contains X dicts corresponding to each one
            of the classes where the model's been trained on.
    
    Returns:
        image (ndarray): Visualization of the results form above.
        
    """
    vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4)
    
    return image


def main():
    # Initialization
    ## Load the configuration variables from 'config.ini'
    config = configparser.ConfigParser()
    config.read('config.ini')
    ## Loading label map
    num_classes = config.getint('net', 'num_classes')
    path_to_labels = config['net']['path_to_labels']
    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, 
        max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
	
	
	
	
	
#previous code

# Loading pre-trained parameters for the cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def detect(image):
    '''
    Function to detect faces/eyes and smiles in the image passed to this function
    '''

    
    image = np.array(image.convert('RGB'))
    
    # Next two lines are for converting the image from 3 channel image (RGB) into 1 channel image
    # img = cv2.cvtColor(new_img, 1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Passing grayscale image to perform detection
    # We pass grayscaled image because opencv expects image with one channel
    # Even if you don't convert the image into one channel, open-cv does it automatically.
    # So, you can just comment line number 26 and 27.
    # If you do, make sure that you change the variables name at appropriate places in the code below
    # Don't blame me if you run into errors while doing that :P
    
    faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)
    # The face_cascade classifier returns coordinates of the area in which the face might be located in the image
    # These coordinates are (x,y,w,h)
    # We will be looking for eyes and smile within this area instead of looking for them in the entire image
    # This makes sense when you're looking for smiles and eyes in a face, if that is not your use case then
    # you can pull the code segment out and make a different function for doing just that, specifically.


    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        
        # The following are the parameters of cv2.rectangle()
        # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
        cv2.rectangle(img=image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
        
        roi = image[y:y+h, x:x+w]
        
        # Detecting eyes in the face(s) detected
        eyes = eye_cascade.detectMultiScale(roi)
        
        # Detecting smiles in the face(s) detected
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
        
        # Drawing rectangle around eyes
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
            
        # Drawing rectangle around smile
        for (sx,sy,sw,sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # Returning the image with bounding boxes drawn on it (in case of detected objects), and faces array
    return image, faces


def about():
	st.write(
		'''
		**Haar Cascade** is an object detection algorithm.
		It can be used to detect objects in images or videos. 

		The algorithm has four stages:

			1. Haar Feature Selection 
			2. Creating  Integral Images
			3. Adaboost Training
			4. Cascading Classifiers



Read more :point_right: https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
https://sites.google.com/site/5kk73gpu2012/assignment/viola-jones-face-detection#TOC-Image-Pyramid
		''')


def main():
    st.title("Face Detection App :sunglasses: ")
    st.write("**Using the Haar cascade Classifiers**")

    activities = ["Home", "About","Object Detection"]
    choice = st.sidebar.selectbox("Pick something fun", activities)

    if choice == "Home":

    	st.write("Go to the About section from the sidebar to learn more about it.")
        
        # You can specify more file types below if you want
    	image_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg', 'webp'])

    	if image_file is not None:

    		image = Image.open(image_file)

    		if st.button("Process"):
                
                # result_img is the image with rectangle drawn on it (in case there are faces detected)
                # result_faces is the array with co-ordinates of bounding box(es)
    			result_img, result_faces = detect(image=image)
    			st.image(result_img, use_column_width = True)
    			st.success("Found {} faces\n".format(len(result_faces)))

    elif choice == "About":
    	about()

	
	
	
	
	#New code
	
	# Streamlit initialization
    st.title("Object Detection")
    st.sidebar.title("Object Detection")
    ## Select classes to be detected by the model
    classes_names = [value['name'] for value in category_index.values()]
    classes_names.sort()
    classes_to_detect = st.sidebar.multiselect(
        "Select which classes to detect", classes_names, ['person'])
    ## Select camera to feed the model
    available_cameras = {'Camera 1': 0, 'Camera 2': 1, 'Camera 3': 2}
    cam_id = st.sidebar.selectbox(
        "Select which camera signal to use", list(available_cameras.keys()))
    ## Select a model to perform the inference
    available_models = [str(i) for i in Path('./trained_model/').iterdir() 
        if i.is_dir() and list(Path(i).glob('*.pb'))]
    model_name = st.sidebar.selectbox(
        "Select which model to use", available_models)
    # Define holder for the processed image
    img_placeholder = st.empty()

    # Model load
    path_to_ckpt = '{}/frozen_inference_graph.pb'.format(model_name)
    detection_graph = model_load_into_memory(path_to_ckpt)

    # Load video source into a thread
    video_source = available_cameras[cam_id]
    ## Start video thread
    video_thread = video_utils.WebcamVideoStream(video_source)
    video_thread.start()
    
    # Detection code
    try:
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while not video_thread.stopped():
                    # Camera detection loop
                    frame = video_thread.read()
                    if frame is None:
                        print("Frame stream interrupted")
                        break
                    # Change color gammut to feed the frame into the network
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    output = run_inference_for_single_image(frame, sess, 
                        detection_graph)
                    output = discriminate_class(output, 
                        classes_to_detect, category_index)
                    processed_image = visualize_results(frame, output, 
                        category_index)

                    # Display the image with the detections in the Streamlit app
                    img_placeholder.image(processed_image)
                    
                    #cv2.imshow('Video', cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break
    
    except KeyboardInterrupt:   
        pass

    print("Ending resources")
    st.text("Camera not detected")
    cv2.destroyAllWindows()
    video_thread.stop()
    sys.exit()


if __name__ == '__main__':
    main()
    
	
	
	
	
	



#if __name__ == "__main__":
   # main()
