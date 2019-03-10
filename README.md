# Lane-Detector
An OpenCV implementation of road lane detection written in Python.

## Method Summary
- Thresholding
- Canny Edge Detection
- Region of Interest Selection
- Hough Transform Line Detection
- Line Averaging

#### Input
I used an image from the [Cityscapes dataset](https://www.cityscapes-dataset.com/).
<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/0.png" width="420" height="230" border="0" /></a> 
</p>

#### Thresholding
Thresholding is possibly a better option than converting an RGB image to HSV or HSL colour space as you're not always guaranteed of lines being an exact white or yellow. 

Threshold is applied using 
```
retval, threshold = cv2.threshold(convertGrayScale(img), 115, 255, cv2.THRESH_BINARY)
```
The parameters supplied to threshold can be tweaked. 115, 255 worked well for me. This gives
<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/1.png" width="420" height="230" border="0" /></a> 
</p>

A gaussian kernel of size 3 is applied to blur out any rough lines and gives
<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/2.png" width="420" height="230" border="0" /></a> 
</p>

#### Canny Edge Detection
<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/3.png" width="420" height="230" border="0" /></a> 
</p>

#### Region of Interest Selection
The lanes, more often than not, fall within the polygon below. The vertices correspond to
```
a = [0, 1023]
b = [0, 630]
c = [720, 455]
d = [1505, 511]
e = [2047, 689]
f = [2047, 1023]
```
<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/7.png" width="420" height="230" border="0" /></a> 
</p>

After filtering the image becomes
<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/4.png" width="420" height="230" border="0" /></a> 
</p>

#### Hough Transform Line Detection
The parameters ho – distance resolution of the accumulator in pixels, theta – angle resolution of the accumulator in radians, threshold - only those lines are returned that get more votes than the threshold, minLineLength – segments shorter than this are rejected, and maxLineGap – maximum allowed gap between points on the same line to link them, can be tweaked to vary output. I used
```
cv2.HoughLinesP(image, rho = 1, theta = np.pi / 180, threshold = 20, minLineLength = 20, maxLineGap = 300)
```

<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/5.png" width="420" height="230" border="0" /></a> 
</p>

#### Line Averaging
Each of the multiple line sets are averaged into one line. To do this I found lines with similar slopes and averaged them into one line. One limitation of this implementation is that it only considers two lanes for final output. However the two lanes can both be left lanes or right lanes unlike other implementations that look for one left and one right lane.

<p align="left">
<img src="https://github.com/acvictor/Lane-Detector/blob/master/images/6.png" width="420" height="230" border="0" /></a> 
</p>


