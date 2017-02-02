##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/test_image.jpg "Distorted"
[image7]: ./camera_cal/test_undist.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to undistort the camera is located in the Udacity Project 4 ipython notebook under the Camera Calibration header. 

I created a matrix that maps all of the 54 corners on the camera calibration chessboard images to a 3 dimensional coordinate to map their position in the real world. I will then call the `findChessboardCorners()` function to find their position in the 2d image plane. These results are appended to the objpoints and imgpoints array for each respective calibration image. These points will be used to compute the the camera calibration and distortion coefficients using the `calibrateCamera()` function. The following images are the results from this calibration.

![alt text][image1]
![alt text][image7]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. You can find the code in the Udacity Project 4 ipython notebook under the Image Detail Extraction header.

I activated pixels on the binary image when they fulfilled the following to pixels:

1) A Sobel operator with a 3x3 kernel on the x-axis orientation with a 30-low 255-high threshold 
2) A HLS threshold operator with a 150-low 255-high threshold

I deactivated pixels on the binary image when they fell outside of the another Sobel operator with a 3x3 kernel on the y-axis orientation with a 3-low 70-high. This combination removed many horizontal line artifacts while maintaining the vertical and curved lane markings.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used the OpenCV `getPerspectiveTransform()` and `warpPerspective()` functions to generate the transformation matrix and apply the transform on the road images. The src and dst points are as follows:

```
src = np.float32([(270, 670),
                  (560, 475),
                  (720, 475),
                  (1020, 670)])

dst = np.float32([(270, 670),
                  (270, 475),
                  (1020, 475),
                  (1020, 670)])

```

Here is the result of the perspective warp.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

