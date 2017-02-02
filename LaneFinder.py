import image_manipulation as img_man
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Finder(object):

    __src = np.float32([(270, 670),
                        (560, 475),
                        (720, 475),
                        (1020, 670)])
    
    __dst = np.float32([(270, 670),
                        (270, 475),
                        (1020, 475),
                        (1020, 670)])


    """Use to find lane markings from video."""
    def __init__(self, camera_cal, src=__src, dst=__dst, force_hist=False):
        super(Finder, self).__init__()
        self.initialized = False
        self.left_fit, self.right_fit = [], []

        self.mtx = camera_cal[0]
        self.dist = camera_cal[1]
        self.src = src
        self.dst = dst
        self.force_hist = force_hist
        

    def discover(self, img):
        undst_img = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        combined_binary = self.img_to_binary(undst_img)
        warped_binary, Minv = self.warp_img(combined_binary)

        if not self.initialized or self.force_hist:
            self.naive_histogram_polyfit(warped_binary)
            self.initialized = True
        else:
            self.fit_lane_line_polynomial(warped_binary)

        left_fit, right_fit = self.left_fit, self.right_fit

        # Generate x and y values for plotting
        fity = np.linspace(0, warped_binary.shape[0] - 1, warped_binary.shape[0] )
        fit_leftx = left_fit[0] * fity ** 2 + left_fit[1] * fity + left_fit[2]
        fit_rightx = right_fit[0] * fity ** 2 + right_fit[1] * fity + right_fit[2]

        ploty = np.linspace(0, 719, num=720) # to cover same y-range as image
        leftx = fit_leftx
        rightx = fit_rightx

        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720 # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + 
                               left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + 
                                right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        # Get meters off road center
        screen_middel_pixel = img.shape[1] / 2
        left_lane_pixel = left_fitx[-1]
        right_lane_pixel = right_fitx[-1]
        car_middle_pixel = int((right_lane_pixel + left_lane_pixel) / 2)
        screen_off_center = screen_middel_pixel - car_middle_pixel
        meters_off_center = xm_per_pix * screen_off_center
        # print(meters_off_center)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 

        # Combine the result with the original image
        undst_img0 = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(undst_img0, "Meters off center: " + str(meters_off_center), 
                    (100,100), font, .6,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(undst_img0, "Turn radius: " + str(left_curverad), 
                    (100,175), font, .6,(255,255,255),2,cv2.LINE_AA)
        result = cv2.addWeighted(undst_img0, 1, newwarp, 0.3, 0)
        return result


    def fit_lane_line_polynomial(self, warped_binary):
        left_fit, right_fit = self.left_fit, self.right_fit

        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "warped_binary")
        # It's now much easier to find line pixels!
        nonzero = warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((
                            nonzerox > (left_fit[0] * 
                                        nonzeroy ** 2 + 
                                        left_fit[1] * nonzeroy + 
                                        left_fit[2] - margin)
                          ) & (
                            nonzerox < (left_fit[0] *
                                         nonzeroy ** 2 + 
                                         left_fit[1] * nonzeroy + 
                                         left_fit[2] + margin)
                          ))

        right_lane_inds = ((
                            nonzerox > (right_fit[0] * 
                                        nonzeroy ** 2 + 
                                        right_fit[1] * nonzeroy + 
                                        right_fit[2] - margin)
                          ) & (
                            nonzerox < (right_fit[0] * 
                                        nonzeroy ** 2 + 
                                        right_fit[1] * nonzeroy + 
                                        right_fit[2] + margin)
                          ))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)


    def naive_histogram_polyfit(self, warped_binary):
        # Assuming you have created a warped binary image called "warped_binary"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(warped_binary[warped_binary.shape[0] / 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(warped_binary.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped_binary.shape[0] - (window + 1) * window_height
            win_y_high = warped_binary.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            
            
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & 
                              (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xleft_low) & 
                              (nonzerox < win_xleft_high)).nonzero()[0]
            
            good_right_inds = ((nonzeroy >= win_y_low) & 
                               (nonzeroy < win_y_high) & 
                               (nonzerox >= win_xright_low) & 
                               (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        
    def img_to_binary(self, img):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sobelx_binary = img_man.get_sobel_binary(rgb_img , orient='x', 
                                         thresh_min=30, thresh_max=255)
        sobely_binary = img_man.get_sobel_binary(rgb_img, orient='y', 
                                         thresh_min=3, thresh_max=70)
        hls_binary = img_man.get_hls_binary(img, thresh_min=150, thresh_max=255)

        combined_binary = np.zeros_like(sobelx_binary)
        combined_binary[(hls_binary == 1) | (sobelx_binary == 1)] = 1
        combined_binary[(sobely_binary == 0)] = 0

        return combined_binary
    
    def warp_img(self, img):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        img_size = (img.shape[1], img.shape[0])
        img_warped = cv2.warpPerspective(img, M, img_size)

        return img_warped, Minv