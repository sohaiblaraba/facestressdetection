# Stress detection system based on face cues

Details to be added soon
![example](https://github.com/sohaib-l/facestressdetection/blob/main/examples/outputdisplay.JPG)

## Requirements
* dlib
* imutils
* opencv
* pillow
* pyautogui

### Installing [dlib](https://github.com/davisking/dlib.git)
In order to have dlib for GPU, you need to install it from source. Assuming you have cmake installed, you need to follow these steps:
```
$ git clone https://github.com/davisking/dlib.git
$ cd dlib
$ mkdir build
$ cd build
$ cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
$ cmake --build .
$ cd ..
$ python setup.py install --yes
$ python setup.py install --set DLIB_USE_CUDA=1 --set USE_AVX_INSTRUCTIONS=1
```

## Usage
### To process a file
```
$ python run.py --input video.mp4
```

### Using the webcam
```
$ python run.py --input 0 --rectangle --landmarks --forehead --forehead_outline --fps
```
* --input 0: specify the input. if input is an integer => webcam. If input is a string => a video path
* --rectangle: to display the rectangle around the detected face. (default: False)
* --landmarks: to display the landmarks of the detected face. (default: False)
* --forehead: to display a color rectangle on the forehead. (defalut: False)
* --forehead_outline: to display the outline around the forehead. (default: False)
* --fps: to show the framerate on the bottom left of the screen. (default: False)

* Press 'q' on the keyboard to quit the program.

