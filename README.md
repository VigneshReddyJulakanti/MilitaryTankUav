# Military Tank Detection and Tracking through UAV

## Problem Statement

The development of autonomous navigation technology for Unmanned Aerial Vehicles (UAVs) is critical in modern warfare. The ability for a UAV to detect, lock-on, and follow the movements of ground-based military tanks and vehicles in real-time is a game-changer. By integrating a high-definition camera and advanced companion computer systems, UAVs can be transformed into formidable reconnaissance and tracking assets on the battlefield. This technology represents a quantum leap in battlefield awareness and situational awareness, providing military commanders with real-time, actionable intelligence in a rapidly changing environment.


## Solution


-Utilizing advanced image processing techniques, our autonomous UAV will detect, lock-on, and track ground-based military tanks and vehicles in real-time.

-The system comprises of a PixHawk 4 flight controller, a Raspberry Pi 4 companion computer, and a high-definition camera.

-The Raspberry Pi camera module V2 streams video data to the companion computer via a USB connection, and the Ground Station receives the video stream via radio waves.

-A comprehensive pre-flight check, conducted through Mission Planner, ensures system functionality and GPS readiness.

-On the Ground Station, a Python script combines YOLOv4 object detection and SORT tracking algorithms to accurately identify and follow the target.

-The Ground Station relays the target's coordinates to the companion computer, which forwards it to the PixHawk flight controller, directing the UAV towards the target.

-The PixHawk PX4 flight controller commands the UAV to maintain a dynamic pursuit of the target, providing real-time situational awareness.


## Dronekit

-We will be utilizing the power of DroneKit, a Python API that leverages the MAVLink protocol to facilitate seamless communication between the flight controller (PixHawk PX4) and the onboard computer (Raspberry Pi or Jetson).

-The onboard computer, equipped with the Raspberry Pi camera module, streams live video footage of the surroundings to the ground station.

-The ground station, running a high-performance machine, employs cutting-edge computer vision techniques such as YOLOv7 and SORT to detect and track military tanks in real-time.

-The latitude and longitude of the detected tanks are then transmitted via telemetry to the onboard computer using the DroneKit API.

-The onboard computer relays the information to the PixHawk PX4 flight controller, which seamlessly navigates the drone towards the target and follows its movements.


## Dataset

-To further boost the accuracy and performance, we have utilized transfer learning on YOLOv4 to fine-tune the model on our custom dataset.

-During inference, we have employed parallel computing using multiple GPUs to speed up the processing time.

-The SORT algorithm is utilized for real-time multiple object tracking to follow the detected tanks as they move.

-The system is designed to be modular and scalable, allowing for easy integration of new sensors and algorithms to further enhance its capabilities.


## Yolo + SORT + Geopy

-To ensure optimal accuracy and precision, we have leveraged cutting-edge deep learning algorithms, such as YOLOv4, to perform real-time detection and tracking of military tanks.

-The SORT algorithm, combined with the powerful geospatial calculations provided by Geopy, allows us to determine the exact location of the tanks, in real-time.

-The DroneKit API, provides a seamless and secure communication between the onboard computer and ground station, allowing for efficient and reliable transmission of tank location data to the drone.

-The system has been designed to be fault-tolerant and is equipped with a return-to-launch feature, ensuring that the drone can safely return to its starting point in the event of any malfunctions or errors.

-By utilizing these advanced technologies, our solution delivers unparalleled accuracy, precision, and reliability for autonomous navigation and tracking of military tanks.



## Applications

-The autonomous navigation system is designed to effectively detect and track moving targets, whether they be enemy military vehicles or wildlife in their natural habitats.

-Utilizing advanced computer vision techniques, such as YOLOv7 object detection and SORT tracking, this system accurately locates and follows the target in real-time.

-With the ability to return to its launch point in case of any operational failures, this system is a reliable tool for military reconnaissance and wildlife monitoring missions.

-By incorporating DroneKit API and integrating with high-performance onboard computing solutions, this system ensures seamless and efficient communication between ground and air components.

## Conclusion

We have been able to:

-Develop a cutting-edge solution for real-time detection and tracking of military tanks and vehicles, as well as wildlife, using state-of-the-art deep learning algorithms and computer vision techniques.

-Implement a robust system architecture with a high-performance onboard computer and flight controller, utilizing DroneKit and MAVLink to ensure seamless communication between the ground station and the drone.

-Achieve remarkable accuracy and reliability in detecting and tracking targets, leveraging a custom-trained YOLOv4 model and the powerful SORT algorithm, allowing for seamless and efficient operations in various scenarios.

-Provide a flexible and scalable platform, capable of adapting to new requirements and improving performance over time, thanks to the use of cutting-edge technologies and techniques.


## Conclusion
The Military Tank Detection and Tracking through UAV project demonstrates the use of advanced computer vision techniques for real-time object detection and tracking. The system can be used for various military surveillance applications.

## Contribution
Feel free to contribute to this project by submitting pull requests or reporting issues.


## Acknowledgements
We would like to acknowledge the following resources for their contributions to this project:
- YOLOv4: https://github.com/AlexeyAB/darknet
- DeepSort: https://github.com/nwojke/deep_sort
- DroneKit: https://dronekit.io/


