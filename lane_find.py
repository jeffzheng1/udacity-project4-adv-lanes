import pickle
import LaneFinder as lf
import imageio
from moviepy.editor import VideoFileClip

undistort_coeff = pickle.load( open( "camera_cal/undistort_pickle.p", "rb" ) )
mtx = undistort_coeff['mtx']
dist = undistort_coeff['dist']
camera_cal = (mtx, dist)
test = lf.Finder(camera_cal)

results = 'results2.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
white_clip = clip1.fl_image(test.discover)
white_clip.write_videofile(results, audio=False)