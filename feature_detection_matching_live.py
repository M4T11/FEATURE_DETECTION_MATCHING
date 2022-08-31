import cv2
import pickle
import os
from timeit import default_timer as timer
from statistics import mean

count = 0

def read_file(filepath):
    with open(filepath, mode='rb') as f:
        data = []
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    kps = []
    dess = []
    descriptions = []
    for value in data:
        index, des, description = value
        kp = []

        for p in index:
            temp = cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=p[1], _angle=p[2],
                                _response=p[3], _octave=p[4], _class_id=p[5])
            kp.append(temp)
        kps.append(kp)
        dess.append(des)
        descriptions.append(description)

        # print(kps)
        # print(dess)
        # print(descriptions)
    return kps, dess, descriptions

def ORB_detector(new_image, image_template, flann=False):
    path = 'movie_ORB_temp'

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    (kp1, des1) = orb.detectAndCompute(image1, None)
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    good_matches = []
    ratio_thresh = 0.8
    if flann:
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1, des2, k=2)

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append([m])

        except ValueError:
            pass

    global count
    if len(good_matches) >= 20:
        count += 1
        img_temp = cv2.drawKeypoints(image_template, kp2, 0, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img3 = cv2.drawMatchesKnn(image1, kp1, img_temp, kp2, good_matches, None, flags=2)
        cv2.imshow('img', img3)
        cv2.imwrite(os.path.join(path, 'ORB_' + str(count) + '.png'), img3)
    return len(good_matches), kp1, good_matches

def ORB_detector_file(new_image, flann=False):
    kps, dess, descriptions = read_file('ORB.pickle')

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()
    (kp1, des1) = orb.detectAndCompute(image1, None)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    ratio_thresh = 0.8
    # Do matching
    for e in range(len(dess)):
        img_to_display = cv2.imread('ORB_ROI_images/' + str(descriptions[e]) + '.png', 0)
        if flann:
            matches = flann_matcher.knnMatch(des1, dess[e], k=2)
        else:
            matches = bf.knnMatch(des1, dess[e], 2)

        good_matches = []
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append([m])

            except ValueError:
                pass

        # print(good_matches)
        # print(descriptions[e])

        if len(good_matches) >= 20:
            img3 = cv2.drawMatchesKnn(image1, kp1, img_to_display, kps[e], good_matches, None, flags=2)
            cv2.putText(img3, descriptions[e] + ', good matches: ' + str(len(good_matches)), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # plt.imshow('img', img3)
            # plt.show()
            cv2.imshow('img' + str(e), img3)

def SIFT_detector(new_image, image_template, flann=False):
    path = 'movie_SIFT_temp'
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    (kp1, des1) = sift.detectAndCompute(image1, None)
    (kp2, des2) = sift.detectAndCompute(image_template, None)
    good_matches = []
    ratio_thresh = 0.8

    if flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1,des2, 2)

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append([m])

        except ValueError:
            pass

    global count
    if len(good_matches) >= 20:
        count+=1
        img_temp = cv2.drawKeypoints(image_template, kp2, 0, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img3 = cv2.drawMatchesKnn(image1, kp1, img_temp, kp2, good_matches, None, flags=2)
        cv2.imshow('img', img3)
        cv2.imwrite(os.path.join(path, 'SIFT_' + str(count) + '.png'), img3)

    return len(good_matches), kp1, good_matches

def SIFT_detector_file(new_image, flann=False):

    kps, dess, descriptions = read_file('SIFT.pickle')

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    (kp1, des1) = sift.detectAndCompute(image1, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    ratio_thresh = 0.8
    # Do matching
    for e in range(len(dess)):
        img_to_display = cv2.imread('SIFT_ROI_images/' + str(descriptions[e]) + '.png', 0)
        if flann:
            matches = flann_matcher.knnMatch(des1, dess[e], k=2)
        else:
            matches = bf.knnMatch(des1,dess[e], 2)

        good_matches = []
        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append([m])

            except ValueError:
                pass

        # print(good_matches)
        # print(descriptions[e])

        if len(good_matches) >= 20:
            img3 = cv2.drawMatchesKnn(image1, kp1, img_to_display, kps[e], good_matches, None, flags=2)
            cv2.putText(img3, descriptions[e] + ', good matches: ' + str(len(good_matches)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # plt.imshow('img', img3)
            # plt.show()
            cv2.imshow('img' + str(e), img3)

def SURF_detector(new_image, image_template, flann=False):
    path = 'movie_SURF_temp'
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()

    (kp1, des1) = surf.detectAndCompute(image1, None)
    (kp2, des2) = surf.detectAndCompute(image_template, None)

    good_matches = []
    ratio_thresh = 0.8
    if flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, 2)

    # Do matching
    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append([m])

        except ValueError:
            pass

    global count
    if len(good_matches) >= 20:
        count+=1
        img_temp = cv2.drawKeypoints(image_template, kp2, 0, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img3 = cv2.drawMatchesKnn(image1, kp1, img_temp, kp2, good_matches, None, flags=2)
        cv2.imshow('img', img3)
        cv2.imwrite(os.path.join(path, 'SURF_' + str(count) + '.png'), img3)
    return len(good_matches), kp1, good_matches

def SURF_detector_file(new_image, flann=False):

    kps, dess, descriptions = read_file('SURF.pickle')

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()

    (kp1, des1) = surf.detectAndCompute(image1, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
    ratio_thresh = 0.8
    for e in range(len(dess)):
        img_to_display = cv2.imread('SURF_ROI_images/' + str(descriptions[e]) + '.png', 0)

        if flann:
            matches = flann_matcher.knnMatch(des1, dess[e], k=2)
        else:
            matches = bf.knnMatch(des1,dess[e], 2)

        good_matches = []

        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append([m])

            except ValueError:
                pass

        # print(good_matches)
        # print(descriptions[e])

        if len(good_matches) >= 20:
            img3 = cv2.drawMatchesKnn(image1, kp1, img_to_display, kps[e], good_matches, None, flags=2)
            cv2.putText(img3, descriptions[e] + ', good matches: ' + str(len(good_matches)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # plt.imshow('img', img3)
            # plt.show()
            cv2.imshow('img' + str(e), img3)

def FAST_SURF_detector(new_image, image_template, flann=False):
    path = 'movie_FAST_SURF_temp'
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)
    surf = cv2.xfeatures2d.SURF_create()
    kp1 = fast.detect(image1, None)
    _, des1 = surf.compute(image1, kp1)

    kp2 = fast.detect(image_template, None)
    _, des2 = surf.compute(image_template, kp2)

    good_matches = []
    ratio_thresh = 0.8

    if flann:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, 2)

    for i, pair in enumerate(matches):
        try:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append([m])

        except ValueError:
            pass
    global count
    if len(good_matches) >= 20:
        count += 1
        img_temp = cv2.drawKeypoints(image_template, kp2, 0, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        img3 = cv2.drawMatchesKnn(image1, kp1, img_temp, kp2, good_matches, None, flags=2)
        cv2.imshow('img', img3)
        cv2.imwrite(os.path.join(path, 'FAST_SURF_' + str(count) + '.png'), img3)
    return len(good_matches), kp1, good_matches


def FAST_SURF_detector_file(new_image, flann=False):

    kps, dess, descriptions = read_file('FAST_SURF.pickle')

    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    fast.setNonmaxSuppression(False)
    surf = cv2.xfeatures2d.SURF_create()
    kp1 = fast.detect(image1, None)
    _, des1 = surf.compute(image1, kp1)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

    # Do matching
    for e in range(len(dess)):
        img_to_display = cv2.imread('FAST_SURF_ROI_images/' + str(descriptions[e]) + '.png', 0)
        if flann:
            matches = flann_matcher.knnMatch(des1, dess[e], k=2)
        else:
            matches = bf.knnMatch(des1,dess[e], 2)

        good_matches = []
        ratio_thresh = 0.8

        for i, pair in enumerate(matches):
            try:
                m, n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append([m])

            except ValueError:
                pass

        # print(good_matches)
        # print(descriptions[e])

        if len(good_matches) >= 20:
            img3 = cv2.drawMatchesKnn(image1, kp1, img_to_display, kps[e], good_matches, None, flags=2)
            cv2.putText(img3, descriptions[e] + ', good matches: ' + str(len(good_matches)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            # plt.imshow('img', img3)
            # plt.show()
            cv2.imshow('img' + str(e), img3)

def choose_algorithm():
    algorithms = {
        "SIFT",
        "SURF",
        "FAST-SURF",
        "ORB",
    }

    choice = input("Select your algorithm: ")
    while choice not in algorithms:
        choice = input(f"Choose one of: {', '.join(algorithms)}: ")

    return choice

def start(path_to_video='fedex.mp4', image_template='fedex.jpg'):
    cap = cv2.VideoCapture(path_to_video)
    image_template = cv2.imread(image_template, 0)
    avg_frame_time = []

    algorithm_choice = choose_algorithm()

    while True:
        ret, frame = cap.read()

        height, width = frame.shape[:2]
        frame_c = frame.copy()

        top_left_x = int(width / 3)
        top_left_y = int((height / 2) + (height / 4))
        bottom_right_x = int((width / 3) * 2)
        bottom_right_y = int((height / 2) - (height / 4))

        cv2.rectangle(frame_c, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,128,255), 2)

        cropped = frame_c[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

        if algorithm_choice == 'SIFT':
            start = timer()
            matches, kp1, good_matches = SIFT_detector(cropped, image_template, True)
            end = timer()
            avg_frame_time.append(end - start)
        elif algorithm_choice == 'SURF':
            start = timer()
            matches, kp1, good_matches = SURF_detector(cropped, image_template, True)
            end = timer()
            avg_frame_time.append(end - start)
        elif algorithm_choice == 'FAST-SURF':
            start = timer()
            matches, kp1, good_matches = FAST_SURF_detector(cropped, image_template, True)
            end = timer()
            avg_frame_time.append(end - start)
        elif algorithm_choice == 'ORB':
            start = timer()
            matches, kp1, good_matches = ORB_detector(cropped, image_template, True)
            end = timer()
            avg_frame_time.append(end - start)

        output_string = "Dopasowania = " + str(matches)
        cv2.putText(frame_c, output_string, (50,450), cv2.FONT_HERSHEY_COMPLEX, 2, (0,128,255), 2)
        threshold = 20

        if matches >= threshold:
            cv2.rectangle(frame_c, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
            cv2.putText(frame_c,'Obiekt zlokalizowany',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)

        cv2.imshow('Test wideo', frame_c)

        if cv2.waitKey(1) == ord('p'):
            frame_copy = frame.copy()
            if algorithm_choice == 'SIFT':
                SIFT_detector_file(frame_copy, True)
            elif algorithm_choice == 'SURF':
                SURF_detector_file(frame_copy, True)
            elif algorithm_choice == 'FAST-SURF':
                FAST_SURF_detector_file(frame_copy, True)
            elif algorithm_choice == 'ORB':
                ORB_detector_file(frame_copy, True)

        if cv2.waitKey(1) == 13: #13 = Enter Key
            break
        print(len(avg_frame_time))
        print(mean(avg_frame_time))

    cap.release()
    cv2.destroyAllWindows()

start()

