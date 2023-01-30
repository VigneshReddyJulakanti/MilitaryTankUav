import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from Custom.coordinates import FramesToCoordinatesAndDistance

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/video/test.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', False, 'dont show video output')
flags.DEFINE_boolean('info', True, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
import time
import math



vehicle = connect('127.0.0.1:14551', wait_ready=True)

intiallon=0
initiallat=0
# Use returned Vehicle object to query device state - e.g. to get the mode:
def distance_two(homeLatitude, homeLongitude, destinationLatitude, destinationLongitude):

    """
    Simple function which returns the distance and bearing between two geographic location

    Inputs:
        1.  homeLatitude            -   Latitude of home location
        2.  homeLongitude           -   Longitude of home location
        3.  destinationLatitude     -   Latitude of Destination
        4.  destinationLongitude    -   Longitude of Destination

    Outputs:
        1. [Distance, Bearing]      -   Distance (in metres) and Bearing angle (in degrees)
                                        between home and destination

    Source:
        https://github.com/TechnicalVillager/distance-bearing-calculation
    """

    rlat1   =   homeLatitude * (math.pi/180) 
    rlat2   =   destinationLatitude * (math.pi/180) 
    rlon1   =   homeLongitude * (math.pi/180) 
    rlon2   =   destinationLongitude * (math.pi/180) 
    dlat    =   (destinationLatitude - homeLatitude) * (math.pi/180)
    dlon    =   (destinationLongitude - homeLongitude) * (math.pi/180)

    # Haversine formula to find distance
    a = (math.sin(dlat/2) * math.sin(dlat/2)) + (math.cos(rlat1) * math.cos(rlat2) * (math.sin(dlon/2) * math.sin(dlon/2)))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    # Distance in metres
    R = 6371e3
    distance = R * c

    # Formula for bearing
   
    
    # Bearing in radians
  
    out = distance

    return out
def change_mode( mode):
        print("Changing to mode: {0}".format(mode))

        vehicle.mode = VehicleMode(mode)
        while vehicle.mode.name != mode:
            print('  ... polled mode: {0}'.format(mode))
            time.sleep(1)

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    # vehicle.mode = VehicleMode("GUIDED")

    while vehicle.mode.name!="GUIDED":
        print("vechicle needs to be changed to guided mode waiting ...")
        time.sleep(1)
        
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.armed = True

    while not vehicle.armed :      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the 
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to 
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation=LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")
        
    return targetlocation


def get_distance_metres(aLocation1, aLocation2):
    """
    Returns the ground distance in metres between two LocationGlobal objects.

    This method is an approximation, and will not be accurate over large distances and close to the 
    earth's poles. It comes from the ArduPilot test code: 
    https://github.com/diydrones/ardupilot/blob/master/Tools/autotest/common.py
    """
    dlat = aLocation2.lat - aLocation1.lat
    dlong = aLocation2.lon - aLocation1.lon
    return math.sqrt((dlat*dlat) + (dlong*dlong)) * 1.113195e5

def goto(dNorth, dEast, gotoFunction=vehicle.simple_goto):
    """
    Moves the vehicle to a position dNorth metres North and dEast metres East of the current position.

    The method takes a function pointer argument with a single `dronekit.lib.LocationGlobal` parameter for 
    the target position. This allows it to be called with different position-setting commands. 
    By default it uses the standard method: dronekit.lib.Vehicle.simple_goto().

    The method reports the distance to target every two seconds.
    """
    
    currentLocation = vehicle.location.global_relative_frame
    targetLocation = get_location_metres(currentLocation, dNorth, dEast)
    targetDistance = get_distance_metres(currentLocation, targetLocation)
    gotoFunction(targetLocation)
    
    #print "DEBUG: targetLocation: %s" % targetLocation
    #print "DEBUG: targetLocation: %s" % targetDistance
    prevRemaining=9999999
    while vehicle.mode.name=="GUIDED": #Stop action if we are no longer in guided mode.
        #print "DEBUG: mode: %s" % vehicle.mode.name
        remainingDistance=get_distance_metres(vehicle.location.global_relative_frame, targetLocation)
        print("Distance to target: ", remainingDistance)
        if remainingDistance>prevRemaining : #Just below target, in case of undershoot.
            print("Reached target")
            break
        prevRemaining=remainingDistance
        time.sleep(1)
    time.sleep(1)



def drone_Frame_relative_goto(yaxis,xaxis,deg):
    yold=xaxis*math.sin(math.radians(deg))+yaxis*math.cos(math.radians(deg))
    xold=xaxis*math.cos(math.radians(deg))-yaxis*math.sin(math.radians(deg))
    goto(yold,xold)
# drone_Frame_relative_goto(10,10)
# deg=315

# drone_Frame_relative_goto(10,-10)

# vehicle.mode = VehicleMode("RTL")
def main(_argv):

    deg=0
    
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0
    
    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    arm_and_takeoff(10)
    intiallon=vehicle.location.global_frame.lon
    initiallat=vehicle.location.global_frame.lat


    frame_num = 0
    # while video is running
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            print('Video has ended or failed, try a different video format!')
            break
        frame_num +=1
        print('Frame #: ', frame_num)
        frame_size = frame.shape[:2]
#         print("frame_size",frame_size)
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['tank']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        # Call the tracker
        tracker.predict()
        tracker.update(detections)
        
        
        #Boom added-----------------------------------------------------------------------
        midx=int(frame_size[1]/2)
        midy=int(frame_size[0]/2)
        drone_track_id=0
        drone_track_age=0
        drone_track_x=0
        drone_track_y=0
        box_mid_y=midy
        box_mid_x=midx
        

        # update tracks
        cntno=0
        for track in tracker.tracks:
            
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            cntno+=1
            if(cntno>=2):
                break
            bbox = track.to_tlbr()
            class_name = track.get_class()
            
        # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
            
            
            #Boom added-------------------------------------------------------------------------
            if(track._max_age>drone_track_age):
                box_mid_x=(int(bbox[0])+int(bbox[2]))/2
                box_mid_y=(int(bbox[1]) +int(bbox[3]))/2
                drone_track_age=track._max_age
                drone_track_id=track.track_id
                drone_track_x=midx-box_mid_x
                drone_track_y=midy-box_mid_y
                
                
        # if enable info flag then print details about each track
            if FLAGS.info:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
#                 midx=int(frame_size[1]/2)
#                 midy=int(frame_size[0]/2)
#                 box_mid_x=(int(bbox[0])+int(bbox[2]))/2
#                 box_mid_y=(int(bbox[1]) +int(bbox[3]))/2
#                 x_axis=""
#                 y_axis=""
#                 if(box_mid_x>midx):
#                     print("move_right ",box_mid_x-midx)
#                     x_axis=f"move_right {box_mid_x-midx}"
                
#                 else:
#                     print("move left ",midx-box_mid_x)
#                     x_axis=f"move left {midx-box_mid_x}"

#                 if(box_mid_y<midy):
#                     print("move up",midy-box_mid_y)
#                     y_axis=f"move up {midy-box_mid_y}"
#                 else:
#                     print("move down",box_mid_y-midy)
#                     y_axis=f"move down {box_mid_y-midy}"

#                 cv2.putText(frame, x_axis,(int(0.8*frame_size[1]), int( 0.1*frame_size[0])),0, 0.75, (255,255,255),2)
#                 cv2.putText(frame, y_axis,(int(0.8*frame_size[1]), int( 0.2*frame_size[0])),0, 0.75, (255,255,255),2)



        if cntno!=0:
        
            #Boom added ------------------------------------------------------------------
            x_axis=""
            y_axis=""
            if(drone_track_x<0):
                print("move_right ",drone_track_x*-1)
                x_axis=f"move_right {drone_track_x*-1}"

            else:
                print("move left ",drone_track_x)
                x_axis=f"move left {drone_track_x}"
            if(drone_track_y>0):
                print("move up",drone_track_y)
                y_axis=f"move up {drone_track_y}"
            else:
                print("move down",drone_track_y*-1)
                y_axis=f"move down {drone_track_y*-1}"

            # cv2.putText(frame, f"tracking object {drone_track_id}",(int(0.8*frame_size[1]), int( 0.05*frame_size[0])),0, 0.75, (255,0,0),2)
            # cv2.putText(frame, x_axis,(int(0.8*frame_size[1]), int( 0.15*frame_size[0])),0, 0.75, (255,0,0),2)
            # cv2.putText(frame, y_axis,(int(0.8*frame_size[1]), int( 0.25*frame_size[0])),0, 0.75, (255,0,0),2)
            cv2.line(frame, (int(box_mid_x),int(box_mid_y)),(int(midx),int(midy)), (255,0,0),1)


             # For now  we are assuming we at these coordinates
            dronePresentCoordinates=[vehicle.location.global_frame.lat,vehicle.location.global_frame.lon]

            # in meters
            presentAltitude=vehicle.location.global_relative_frame.alt

            temp=FramesToCoordinatesAndDistance(dronePresentCoordinates,[midx,midy],[box_mid_x,box_mid_y],presentAltitude,[frame_size[1],frame_size[0]])

            if distance_two(initiallat,intiallon,temp["newLatitude"],temp["newLongitude"])<10 :

            

                drone_Frame_relative_goto(temp['dist_y_meters'],temp['dist_x_meters'],vehicle.)

                deg+=temp['bearing']%360

                # for printing on image
                tempstr=f"angle with north : {temp['bearing']}, shortest distance : {temp['dist']} meters "
                cv2.putText(frame, tempstr,(int(0.01*frame_size[1]),  int( 0.05*frame_size[0])),0, 0.75, (255,0,0),2)

                tempstr=f"dist_x_meters: {temp['dist_x_meters']}, dist_y_meters: {temp['dist_y_meters']}"
                cv2.putText(frame, tempstr,(int(0.01*frame_size[1]),  int( 0.15*frame_size[0])),0, 0.75, (255,0,0),2)

                tempstr=f"newLongitude: {temp['newLongitude']}, newLatitude:  {temp['newLatitude']}"
                cv2.putText(frame, tempstr,(int(0.01*frame_size[1]),  int( 0.25*frame_size[0])),0, 0.75, (255,0,0),2)
            else:
                print("new coordinates out of geo fence")
            
        else:
            print("tank not detected in the frame")
        
        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
#         print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)
        
        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): 

            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        change_mode("RTL")
