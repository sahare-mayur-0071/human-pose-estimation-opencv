# Human Pose Estimation using OpenPose and OpenCV

This project was developed as part of an **Internship Program under AICET (All India Council for Education & Training)**.  
The objective of this project is to implement **Human Pose Estimation** using **OpenPose (MobileNet model)** and **OpenCV** in Python.

---

## Internship Details
- **Internship Organization:** AICET  
- **Project Title:** Human Pose Estimation using OpenPose and OpenCV  
- **Domain:** Computer Vision / Artificial Intelligence  
- **Duration:** Internship Project  
- **Purpose:** Academic & Internship Submission  

---

## Project Description
Human Pose Estimation is a computer vision technique used to detect human body keypoints such as head, arms, legs, and torso.  
This project detects **18 body keypoints** from images or live webcam input and visualizes them using a skeleton structure.

---

## Features
- Detection of human body joints
- Skeleton visualization using lines and points
- Supports image and webcam input
- Fast and lightweight OpenPose MobileNet model
- Easy-to-understand Python implementation

---

## Technologies Used
- Python
- OpenCV
- NumPy
- OpenPose (MobileNet)

---

## Project Structure
human-pose-estimation-opencv/
│── openpose.py
│── graph_opt.pb
│── image.jpg
│── output.JPG
│── README.md
│── .gitignore

---

## Installation

1. Clone the repository
git clone https://github.com/sahare-mayur-0071/human-pose-estimation-opencv.git

cd human-pose-estimation-opencv


2. Install dependencies


pip install opencv-python numpy


---

## How to Run

### Run using webcam


python openpose.py


### Run using image


python openpose.py --input image.jpg


### Adjust confidence threshold


python openpose.py --input image.jpg --thr 0.5


---

## Output
The system detects human body joints and displays the pose skeleton.

![Output](output.JPG)

---

## Model File
This project uses the **OpenPose MobileNet model**:
- `graph_opt.pb`

Ensure the model file is present in the root directory before execution.

---

## Learning Outcomes
- Understanding of human pose estimation
- Hands-on experience with OpenCV
- Practical exposure to AI and computer vision
- Model integration in Python applications

---

## Declaration
This project is developed **solely for educational and internship purposes under AICET**.  
All resources used are for learning and non-commercial use.

---

## Acknowledgement
I would like to thank **AICET** for providing the internship opportunity and guidance for this project.
