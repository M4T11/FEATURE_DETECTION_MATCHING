import cv2 as cv
from timeit import default_timer as timer

def sift_algorithm(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('sift_keypoints.jpg',img)
    return keypoints, descriptors

def sift_algorithm_timer(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.xfeatures2d.SIFT_create()
    start_keypoints = timer()
    keypoints = sift.detect(gray, None)
    end_keypoints = timer()
    start_descriptors = timer()
    _, descriptors = sift.compute(gray, keypoints)
    end_descriptors = timer()
    time_keypoints = end_keypoints - start_keypoints
    time_descriptors = end_descriptors - start_descriptors
    # print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('sift_keypoints.jpg',img)
    return keypoints, descriptors, time_keypoints, time_descriptors

def surf_algorithm(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(gray, None)
    print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('surf_keypoints.jpg',img)
    return keypoints, descriptors

def surf_algorithm_timer(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    surf = cv.xfeatures2d.SURF_create()
    start_keypoints = timer()
    keypoints = surf.detect(gray, None)
    end_keypoints = timer()
    start_descriptors = timer()
    _, descriptors = surf.compute(gray, keypoints)
    end_descriptors = timer()
    time_keypoints = end_keypoints - start_keypoints
    time_descriptors = end_descriptors - start_descriptors
    # print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('sift_keypoints.jpg',img)
    return keypoints, descriptors, time_keypoints, time_descriptors

def fast_surf_algorithm(img, NonmaxSuppression=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    fast.setNonmaxSuppression(NonmaxSuppression)
    surf = cv.xfeatures2d.SURF_create()
    keypoints = fast.detect(gray, None)
    _, descriptors = surf.compute(gray, keypoints)
    print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('fast_keypoints.jpg',img)
    return keypoints, descriptors

def fast_surf_algorithm_timer(img, NonmaxSuppression=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    fast.setNonmaxSuppression(NonmaxSuppression)
    surf = cv.xfeatures2d.SURF_create()
    start_keypoints = timer()
    keypoints = fast.detect(gray, None)
    end_keypoints = timer()
    start_descriptors = timer()
    _, descriptors = surf.compute(gray, keypoints)
    end_descriptors = timer()
    time_keypoints = end_keypoints - start_keypoints
    time_descriptors = end_descriptors - start_descriptors
    # print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('fast_keypoints.jpg',img)
    return keypoints, descriptors, time_keypoints, time_descriptors

def fast_brief_algorithm(img, NonmaxSuppression=False):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    fast.setNonmaxSuppression(NonmaxSuppression)
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints = fast.detect(gray, None)
    _, descriptors = brief.compute(gray, keypoints)
    print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('fast_keypoints.jpg',img)
    return keypoints, descriptors

def orb_algorithm(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    print('Number of keypoints:', len(keypoints))
    return keypoints, descriptors

def orb_algorithm_timer(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    orb = cv.ORB_create()
    start_keypoints = timer()
    keypoints = orb.detect(gray, None)
    end_keypoints = timer()
    start_descriptors = timer()
    _, descriptors = orb.compute(gray, keypoints)
    end_descriptors = timer()
    time_keypoints = end_keypoints - start_keypoints
    time_descriptors = end_descriptors - start_descriptors
    # print('Number of keypoints:', len(keypoints))
    # img = cv.drawKeypoints(gray,keypoints,img)
    # cv.imwrite('orb_keypoints.jpg',img)
    return keypoints, descriptors, time_keypoints, time_descriptors
