# Stress detection system based on face cues

Details to be added soon


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
Specify the id of the webcam. Generally 0 for the integrated webcam.
```
$ python run.py --input 0
```
