## Writeup

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

[pipeline_original]: ./output_images/pipeline_original.jpg 
[pipeline_transformed]: ./output_images/pipeline_transformed.jpg 
[perspective_original]: ./output_images/perspective_original.jpg
[perspective_transformed]: ./output_images/perspective_transformed.jpg
[color_original]: ./output_images/color_original.jpg 
[color_transformed]: ./output_images/color_transformed.jpg 
[undist_original]: ./output_images/undist_original.jpg 
[undist_transformed]: ./output_images/undist_transformed.jpg 
[road_image_original]: ./output_images/road_image_original.jpg 
[road_image_undist]: ./output_images/road_image_undist.jpg 
[video1]: ./project_video.mp4 

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In order to correctly detect image features and measure different distances, we should consider the image calibration and undistortion.

We can leverage the chessboard image to prepare calibration coefficients for the future undistortions.

1. We start by preparring `object points` which are 3d point in real world space.
2. Then, we iterate our images, convert them to `grayscale` and try to find the chessboard corners:

    ```python
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, imgpoint = cv2.findChessboardCorners(gray_img, (nx, ny), None)
    ```

3. If `image points` were found (2d points in image plane), we collect both `image points and object points` for the future camera callibration.

4. After thar we can use our collected points to callibrate the camera and undistort some of the images:

    ```python
    _, mat, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)
    undist = cv2.undistort(image, mat, dist, None, mat)
    ```

This will result in the following:

1. Original image:
    ![alt text][undist_original]


1. Undistorted image:
    ![alt text][undist_transformed]


Road undistortion example:

1. Original image:
    ![alt text][road_image_original]


1. Undistorted image:
    ![alt text][road_image_undist]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

[The code described in this part is located here](./Advanced_Lanes.ipynb#Color-Transforms-and-Gradients) 

I experimented with multiple combinations of the different thresholds:

1. x_sobel_threshold
2. y_sobel_threshold
3. s_channel_threshold
4. direction_threshold
5. magnitude_threshold

The best combination I found was the following:

```
magnitude_threshold | (x_sobel_threshold & y_sobel_threshold)
```

First I found both x_sobel and y_sobel by applying `Sobel` function:

```python
x_sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
y_sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
```

Then, in order to find the `magnitude threshold` we calculate the sqaure root of sum of sqaured x_sobel_threshold and sqaured y_sobel_threshold:

```python
magnitude = np.sqrt(x_sobel**2 + y_sobel**2)
```

Finally, we can get the combined binary out of these threshold like this:

```python
combined_binary = np.zeros_like(direction_threshold)
combined_binary[(magnitude_threshold == 1) 
                | ((x_threshold == 1) & (y_threshold == 1))] = 1
```

The above steps resulted the binary image where we can easily distingue the road lanes:

1. Original image:
    ![alt text][color_original]


1. Color transformed image:
    ![alt text][color_transformed]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

[The code described in this part is located here](./Advanced_Lanes.ipynb#Perspective-Transform) 

First of all we need to define 2 matrices for `source` and `destination` images:

```python
height = image.shape[0]
width = image.shape[1]

src = np.float32([[width // 2 - 80, height * 0.625], 
                  [width // 2 + 80, height * 0.625], 
                  [-80, height], 
                  [width + 80, height]])
dst = np.float32([[80, 0], 
                  [width - 80, 0], 
                  [80, height], 
                  [width - 80, height]])
```

Then, I applied the `getPerspectiveTransform` to extract the perspective transformation matrix `M`

After, we can use `warpPerspective(image, M, image_size)` function to get our `warped image`:

1. Original image:
    ![alt text][perspective_original]


1. Warped (Perspective Transformed) image:
    ![alt text][perspective_transformed]


Also, I calcualte and return `unwarped_m` by applying `getPerspectiveTransform` method but with swaped `dst and src`:

```python
unwarped_m = cv2.getPerspectiveTransform(dst, src)
```

`unwarped_m` allows us to transform the image back to the original once we performed all the required calculations.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This step actually happens in 2 different classes:

1. The first class SlidingWindow [SlidingWindow](./Advanced_Lanes.ipynb#Sliding-Window) takes care of the lane-line pixels identification.
It uses collected window indecies in order to keep only `road lane pixles`:

    ```python
    win_inds = ((nonzero[0] >= self.y_high) & 
                (nonzero[0] < self.y_low) & 
                (nonzero[1] >= self.x_center - self.margin) &
                (nonzero[1] < self.x_center + self.margin)).nonzero()[0]

    if len(win_inds) > self.min_pix:
        self.x_mean = np.int64(np.mean(nonzero[1][win_inds]))
        
    else:
        self.x_mean = self.x_center
    ```

    Also, it calculates the `x_mean` of the current window. This variable is getting passed to the next window `x_center` aka `line base`.

2. The second class [RoadLane](./Advanced_Lanes.ipynb#Road-Lane) fits lanes positions with the second degree polynomial:

    ```python
    y = np.linspace(0, self.heigth - 1, self.heigth)
    poly_fit = np.array(self.points).mean(axis=0)

    return np.stack((poly_fit[0] * y ** 2 + poly_fit[1] * y + poly_fit[2], y)).astype(np.int).T
    ```


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The [RoadLane](./Advanced_Lanes.ipynb#Road-Lane) class also takes care of curvature and position calcualtions:

1. In order to find the curvative of the road, I calcualte the curvative of each lane first:

    ```python
    fit_curvature = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

    radius = ( ( (1 + (2 * fit_curvature[0] * y_eval_const * ym_per_pix + fit_curvature[1])**2) **1.5 ) 
                / np.absolute(2 * fit_curvature[0]))
    ```

    Moreover, I also scale the image pixels to the real world road dimensions using these constants described in the course lesons:

    ```python
    ym_per_pix = 30 / 720 
    xm_per_pix = 3.7 / 700
    ```

    Once we get both lanes curvative, we find the road curvative by using this functions in the main [Pipeline](./Advanced_Lanes.ipynb#Pipeline) class:

    ```python
    def calculate_road_radius(self, left_lane, right_lane):
            
            # getting mean between two lanes curvative
            return np.average([left_lane.calculate_lane_radius(), 
                            right_lane.calculate_lane_radius()])
    ```

2. It was relatively easy to find the car position:

    1. Get the `max x point`
    2. Get the `middle of the road` and subtract the `max x point`
    3. Multiply by real world scale const `xm_per_pix`

    ```python
    xm_per_pix = 3.7 / 700 
        
    points = self.get_points()
    x = points[np.max(points[:,1])][0]
    
    return np.absolute((self.width // 2 - x) * xm_per_pix)
    ```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Finally, all the above classes are combined together in the main [Pipeline](./Advanced_Lanes.ipynb#Pipeline) class. 

After calculating the road lanes parameters, the function `display_lane` uses `unwarped_m` to transform the road overlay back to the original image resulting nice color overlap on the detect road. In addition, both `radius` and `center` values are added as text to the image.

![alt text][pipeline_transformed]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my pipeline video result](./output_video/output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Of course, even if this project is considered Advanced Lane detection, it is still has a lot of drawbacks and not good enough for the production self-driving car. 

1. The first major issue I identified is the big shadow that came up in the `challenge video`. Even if we applied multiple thresholds it is still really hard to eliminate all the shadows and keep all the lane pixels. Therefore, the pipeline still needs improvement in the effective shadow elimination. 

2. Also, if we run the current pipeline on the `harder challenge video`, we will see that it really struggles to capture the road lanes. This is due to the extreme road curvative, excessive trees' shadows, and light. 
