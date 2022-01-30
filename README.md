# OUR_EYE_ASSESSMENT_PERSON_DETECTOR

Detects person in image, video and through webcam.

# To use it:

- Requirements: windows with docker

- Clone repo in your working directory

- Build docker image:(replace D:\Our_Eye_Task with your folder)

> docker build -t oureye D:\Our_Eye_Task .

- Configure exec.sh script (see bellow) 

- Run Image:(replace D:/Our_Eye_Task with your folder)

> docker run -v D:/Our_Eye_Task:/oureyeapp oureye 



# To configure it:

Configuration is made in exec.sh at python function call:

> python OurEye.ai_Assessment_Person_Detection.py ...

All possible arguments are:

```
-t (--inputType): type=int, default=1: # 0 for detection on img, 1 for detection on video, 2 for detection through webcam(#will work only when docker run in virtualbox with window/mac and on linux use --device=/dev/video0:/dev/video0)")

-i (--input), type=str, default="videoplayback.mp4": complete name of input file with .ext, don't use when -type = 2

-o (--output), type=str, default="Labeled_IMG.png"(for -t = 0), "Labeled_Video.mp4"(for -t = 1), "Labeled_Video_RealTime.mp4"(for -t = 2)  : "complete name of output file with .ext, #incase of video add .mp4 and image add .png ")

```
#### Suggested configuration (exec.sh):

- Detection on video: default values

> python OurEye.ai_Assessment_Person_Detection.py -t 1 -i videoplayback.mp4 -o Labeled_Video.mp4

- Detection on img: default values

> python OurEye.ai_Assessment_Person_Detection.py -t 0 -i sampleImg.png -o Labeled_IMG.png

- Detection through webcam: default values

> python OurEye.ai_Assessment_Person_Detection.py -t 2 

#### Input/Ouput files

- Inputs file are in input/ folder

- Outputs file are in output/ folder (.mp4,.png)

#### log folder

- debug file contains logs for program execution. Use it to debug in case of any error.

- App_Info file contains total number of person detected for each frame.

# Tools versions:

- python 3.8
- pytorch 1.10.1
- opencv-python-headless 4.5.5.62
- Model YOLOv5l

# OS compatibility:

This project is developed on windows. For windows/mac it seems impossible to use webcam from docker(--inputType = 2) directly(cv2.VideoCapture(0) gives following error: "VIDEOIO(V4L2:/dev/video0): can't open camera by index"), as macOS and Windows do not have the path: /dev/video0. Only solution that I can find was running docker inside the virtualbox and change the default virtual to use webcam. See [https://medium.com/@jijupax/connect-the-webcam-to-docker-on-mac-or-windows-51d894c44468] for more information).
 

# Remarks for running it on linux:

For linux building dockerfile this way will work, but for VideoCapture(0) to work we can't use pip to install opencv, opencv need to be installed directly. And while running the image on linux we need to pass "docker run --device=/dev/video0". I have added the sample Dockerfile but didn't have linux os to test it.

# Runing without docker:

Program can be run without using docker by using environment.yml file.
> conda env create -f person_detector.yml
> python OurEye.ai_Assessment_Person_Detection.py -t 2