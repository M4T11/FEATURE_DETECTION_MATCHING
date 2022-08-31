import cv2 as cv
from os import listdir
from os.path import isfile, join
import algorithms
import matching_methods
from timeit import default_timer as timer
import numpy as np

def test_img(img_set):

    images = [f for f in listdir(img_set) if isfile(join(img_set, f))]
    # Time in seconds
    for image in images:
        img = cv.imread(img_set+'/'+image)

        sift_keypoints, sift_descriptors, sift_time_keypoints, sift_time_descriptors = algorithms.sift_algorithm_timer(img)
        sift_time_per_keypoint = sift_time_keypoints/len(sift_keypoints)
        sift_time_per_descriptor = sift_time_descriptors / len(sift_descriptors)
        surf_keypoints, surf_descriptors, surf_time_keypoints, surf_time_descriptors = algorithms.surf_algorithm_timer(img)
        surf_time_per_keypoint = surf_time_keypoints/len(surf_keypoints)
        surf_time_per_descriptor = surf_time_descriptors / len(surf_descriptors)
        fast_surf_keypoints, fast_surf_descriptors, fast_surf_time_keypoints, fast_surf_time_descriptors = algorithms.fast_surf_algorithm_timer(img)
        fast_surf_time_per_keypoint = fast_surf_time_keypoints/len(fast_surf_keypoints)
        fast_surf_time_per_descriptor = fast_surf_time_descriptors / len(fast_surf_descriptors)
        orb_keypoints, orb_descriptors, orb_time_keypoints, orb_time_descriptors = algorithms.orb_algorithm_timer(img)
        orb_time_per_keypoint = orb_time_keypoints/len(orb_keypoints)
        orb_time_per_descriptor = orb_time_descriptors / len(orb_descriptors)

        with open(img_set + "_results.txt", 'a') as f:
            f.write(str(image) + "\n" + "\n" + "\n")
            f.write("SIFT -> ilosc deskryptorow zdefiniowanych dla obiektu: " + str(
                len(sift_descriptors)) + ", czas wykrycia pojedynczego punktu charakterystycznego: " + str("{:.10f}".format(
                sift_time_per_keypoint)) + "s, czas utworzenia deskryptora pojedynczego punktu charakterystycznego: " + str(
                "{:.10f}".format(sift_time_per_descriptor)) + "s, calkowity czas wygenerowania deskryptorow: " + str(
                sift_time_descriptors) + "s, calkowity czas wygenerowania punktow charakterystycznych: " + str(
                sift_time_keypoints) + "s, calkowity czas dzialania: " + str(
                sift_time_descriptors+sift_time_keypoints) + "\n")
            f.write("SURF -> ilosc deskryptorow zdefiniowanych dla obiektu: " + str(
                len(surf_descriptors)) + ", czas wykrycia pojedynczego punktu charakterystycznego: " + str("{:.10f}".format(
                surf_time_per_keypoint)) + "s, czas utworzenia deskryptora pojedynczego punktu charakterystycznego: " + str(
                "{:.10f}".format(surf_time_per_descriptor)) + "s, calkowity czas wygenerowania deskryptorow: " + str(
                surf_time_descriptors) + "s, calkowity czas wygenerowania punktow charakterystycznych: " + str(
                surf_time_keypoints) + "s, calkowity czas dzialania: " + str(
                surf_time_descriptors+surf_time_keypoints) + "\n")
            f.write("FAST_SURF -> ilosc deskryptorow zdefiniowanych dla obiektu: " + str(
                len(fast_surf_descriptors)) + ", czas wykrycia pojedynczego punktu charakterystycznego: " + str("{:.10f}".format(
                fast_surf_time_per_keypoint)) + "s, czas utworzenia deskryptora pojedynczego punktu charakterystycznego: " + str(
                "{:.10f}".format(fast_surf_time_per_descriptor)) + "s, calkowity czas wygenerowania deskryptorow: " + str(
                fast_surf_time_descriptors) + "s, calkowity czas wygenerowania punktow charakterystycznych: " + str(
                fast_surf_time_keypoints) + "s, calkowity czas dzialania: " + str(
                fast_surf_time_descriptors+fast_surf_time_keypoints) + "\n")
            f.write("ORB -> ilosc deskryptorow zdefiniowanych dla obiektu: " + str(
                len(orb_descriptors)) + ", czas wykrycia pojedynczego punktu charakterystycznego: " + str("{:.10f}".format(
                orb_time_per_keypoint)) + "s, czas utworzenia deskryptora pojedynczego punktu charakterystycznego: " + str(
                "{:.10f}".format(orb_time_per_descriptor)) + "s, calkowity czas wygenerowania deskryptorow: " + str(
                orb_time_descriptors) + "s, calkowity czas wygenerowania punktow charakterystycznych: " + str(
                orb_time_keypoints) + "s, calkowity czas dzialania: " + str(
                orb_time_descriptors+orb_time_keypoints) + "\n")
            f.write("\n" + "\n" + "\n")


def test_img_matching(img_1_path, img_2_path, matching_algorithm):
    img_1 = cv.imread(img_1_path)
    # cv.imshow("img_1", img_1)
    img_2 = cv.imread(img_2_path)
    # cv.imshow("img_2", img_2)

    sift_keypoints_1, sift_descriptors_1, sift_time_keypoints_1, sift_time_descriptors_1 = algorithms.sift_algorithm_timer(img_1)
    sift_keypoints_2, sift_descriptors_2, sift_time_keypoints_2, sift_time_descriptors_2 = algorithms.sift_algorithm_timer(img_2)

    surf_keypoints_1, surf_descriptors_1, surf_time_keypoints_1, surf_time_descriptors_1 = algorithms.surf_algorithm_timer(img_1)
    surf_keypoints_2, surf_descriptors_2, surf_time_keypoints_2, surf_time_descriptors_2 = algorithms.surf_algorithm_timer(img_2)

    fast_surf_keypoints_1, fast_surf_descriptors_1, fast_surf_time_keypoints_1, fast_surf_time_descriptors_1 = algorithms.fast_surf_algorithm_timer(img_1)
    fast_surf_keypoints_2, fast_surf_descriptors_2, fast_surf_time_keypoints_2, fast_surf_time_descriptors_2 = algorithms.fast_surf_algorithm_timer(img_2)

    orb_keypoints_1, orb_descriptors_1, orb_time_keypoints_1, orb_time_descriptors_1 = algorithms.orb_algorithm_timer(img_1)
    orb_keypoints_2, orb_descriptors_2, orb_time_keypoints_2, orb_time_descriptors_2 = algorithms.orb_algorithm_timer(img_2)

    good_matches_rate_sift = None
    good_matches_rate_surf = None
    good_matches_rate_fast_surf = None
    good_matches_rate_orb = None
    sift_time = None
    surf_time = None
    fast_surf_time = None
    orb_time = None

    if matching_algorithm == 'BRUTE-FORCE':
        print("BRUTE-FORCE")
        start_sift = timer()
        good_sift, matches_sift = matching_methods.brute_force_match(sift_descriptors_1, sift_descriptors_2, ratio=True)
        print(len(good_sift))
        print(len(matches_sift))
        end_sift = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, sift_keypoints_1, img_2, sift_keypoints_2, good_sift, None)
        # cv.imshow("Matching Images1", matched_imgs)
        # print((len(good) / max(len(sift_keypoints_1), len(sift_keypoints_2))) * 100)
        print((len(good_sift)/len(matches_sift))*100)
        good_matches_rate_sift = (len(good_sift) / len(sift_descriptors_1)) * 100

        start_surf = timer()
        good_surf, matches_surf = matching_methods.brute_force_match(surf_descriptors_1, surf_descriptors_2, ratio=True)
        end_surf = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, surf_keypoints_1, img_2, surf_keypoints_2, good_surf, None)

        good_matches_rate_surf = (len(good_surf)/len(surf_descriptors_1))*100
        print((len(good_surf) / len(matches_surf)) * 100)
        # cv.imshow("Matching Images2", matched_imgs)
        start_fast_surf = timer()
        good_fast_surf, matches_fast_surf = matching_methods.brute_force_match(fast_surf_descriptors_1, fast_surf_descriptors_2, ratio=True)
        end_fast_surf = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, fast_surf_keypoints_1, img_2, fast_surf_keypoints_2, good_fast_surf, None)

        print((len(good_fast_surf)/len(matches_fast_surf))*100)
        good_matches_rate_fast_surf = (len(good_fast_surf) / len(fast_surf_descriptors_1)) * 100

        # cv.imshow("Matching Images3", matched_imgs)
        start_orb = timer()
        good_orb, matches_orb = matching_methods.brute_force_match(orb_descriptors_1, orb_descriptors_2, ratio=True, orb=True)
        end_orb = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, orb_keypoints_1, img_2, orb_keypoints_2, good_orb, None)

        print((len(good_orb)/len(matches_orb))*100)
        good_matches_rate_orb = (len(good_orb) / len(orb_descriptors_1)) * 100
        # cv.imwrite('ORB_matched.png', matched_imgs)

        # cv.imshow("Matching Images4", matched_imgs)

        # cv.waitKey(0)
        sift_time = end_sift - start_sift
        surf_time = end_surf - start_surf
        fast_surf_time = end_fast_surf - start_fast_surf
        orb_time = end_orb - start_orb
        print(sift_time)
        print(surf_time)
        print(fast_surf_time)
        print(orb_time)

    elif matching_algorithm == 'FLANN':
        print("FLANN")
        start_sift = timer()
        good_sift, matches_sift = matching_methods.FLANN_match(sift_descriptors_1, sift_descriptors_2)
        print(len(good_sift))
        print(len(matches_sift))
        end_sift = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, sift_keypoints_1, img_2, sift_keypoints_2, good_sift, None)
        # cv.imshow("Matching Images1", matched_imgs)
        # print((len(good) / max(len(sift_keypoints_1), len(sift_keypoints_2))) * 100)
        print((len(good_sift) / len(matches_sift)) * 100)
        good_matches_rate_sift = (len(good_sift) / len(sift_descriptors_1)) * 100

        start_surf = timer()
        good_surf, matches_surf = matching_methods.FLANN_match(surf_descriptors_1, surf_descriptors_2)
        end_surf = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, surf_keypoints_1, img_2, surf_keypoints_2, good_surf, None)

        good_matches_rate_surf = (len(good_surf) / len(surf_descriptors_1)) * 100
        print((len(good_surf) / len(matches_surf)) * 100)
        # cv.imshow("Matching Images2", matched_imgs)
        start_fast_surf = timer()
        good_fast_surf, matches_fast_surf = matching_methods.FLANN_match(fast_surf_descriptors_1, fast_surf_descriptors_2)
        end_fast_surf = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, fast_surf_keypoints_1, img_2, fast_surf_keypoints_2, good_fast_surf, None)

        print((len(good_fast_surf) / len(matches_fast_surf)) * 100)
        good_matches_rate_fast_surf = (len(good_fast_surf) / len(fast_surf_descriptors_1)) * 100

        # cv.imshow("Matching Images3", matched_imgs)
        start_orb = timer()
        good_orb, matches_orb = matching_methods.FLANN_match(orb_descriptors_1, orb_descriptors_2, orb=True)
        end_orb = timer()
        matched_imgs = cv.drawMatchesKnn(img_1, orb_keypoints_1, img_2, orb_keypoints_2, good_orb, None)

        print((len(good_orb) / len(matches_orb)) * 100)
        good_matches_rate_orb = (len(good_orb) / len(orb_descriptors_1)) * 100
        # cv.imwrite('ORB_matched_FLANN.png', matched_imgs)
        # cv.imshow("Matching Images4", matched_imgs)

        # cv.waitKey(0)
        sift_time = end_sift - start_sift
        surf_time = end_surf - start_surf
        fast_surf_time = end_fast_surf - start_fast_surf
        orb_time = end_orb - start_orb
        print(sift_time)
        print(surf_time)
        print(fast_surf_time)
        print(orb_time)

    with open(matching_algorithm + "_matching_results.txt", 'a') as f:
        f.write(str(img_1_path) + " " + str(img_2_path) +"\n" + "\n")
        f.write("SIFT -> ilosc dopasowan: " + str(
            len(good_sift)) + ", procent dopasowan: " + str(
            good_matches_rate_sift) + ", calkowity czas dopasowania: " + str("{:.10f}".format(sift_time))+ "\n")
        f.write("SURF -> ilosc dopasowan: " + str(
            len(good_surf)) + ", procent dopasowan: " + str(
            good_matches_rate_surf) + ", calkowity czas dopasowania: " + str("{:.10f}".format(surf_time))+ "\n")
        f.write("FAST_SURF -> ilosc dopasowan: " + str(
            len(good_fast_surf)) + ", procent dopasowan: " + str(
            good_matches_rate_fast_surf) + ", calkowity czas dopasowania: " + str("{:.10f}".format(fast_surf_time)) + "\n")
        f.write("ORB -> ilosc dopasowan: " + str(
            len(good_orb)) + ", procent dopasowan: " + str(
            good_matches_rate_orb) + ", calkowity czas dopasowania: " + str("{:.10f}".format(orb_time)) + "\n")
        f.write("\n" + "\n")
    # cv.waitKey(0)

def group_of_descriptors(img_to_find_path, algorithm, matching_method):
    img_corner_1 = cv.imread('HOUSE/HD/Corners/Corner_1.jpg')
    img_corner_2 = cv.imread('HOUSE/HD/Corners/Corner_2.jpg')
    img_corner_3 = cv.imread('HOUSE/HD/Corners/Corner_3.jpg')
    img_corner_4 = cv.imread('HOUSE/HD/Corners/Corner_4.jpg')

    img_to_find = cv.imread(img_to_find_path)

    if algorithm == 'SIFT':
        print('SIFT')
        keypoints_corner_1, descriptors_corner_1 = algorithms.sift_algorithm(img_corner_1)
        keypoints_corner_2, descriptors_corner_2 = algorithms.sift_algorithm(img_corner_2)
        keypoints_corner_3, descriptors_corner_3 = algorithms.sift_algorithm(img_corner_3)
        keypoints_corner_4, descriptors_corner_4 = algorithms.sift_algorithm(img_corner_4)

        keypoints_to_find, descriptors_to_find = algorithms.sift_algorithm(img_to_find)
    elif algorithm == 'SURF':
        print('SURF')
        keypoints_corner_1, descriptors_corner_1 = algorithms.surf_algorithm(img_corner_1)
        keypoints_corner_2, descriptors_corner_2 = algorithms.surf_algorithm(img_corner_2)
        keypoints_corner_3, descriptors_corner_3 = algorithms.surf_algorithm(img_corner_3)
        keypoints_corner_4, descriptors_corner_4 = algorithms.surf_algorithm(img_corner_4)

        keypoints_to_find, descriptors_to_find = algorithms.surf_algorithm(img_to_find)
    elif algorithm == 'FAST_SURF':
        print('FAST_SURF')
        keypoints_corner_1, descriptors_corner_1 = algorithms.fast_surf_algorithm(img_corner_1)
        keypoints_corner_2, descriptors_corner_2 = algorithms.fast_surf_algorithm(img_corner_2)
        keypoints_corner_3, descriptors_corner_3 = algorithms.fast_surf_algorithm(img_corner_3)
        keypoints_corner_4, descriptors_corner_4 = algorithms.fast_surf_algorithm(img_corner_4)

        keypoints_to_find, descriptors_to_find = algorithms.fast_surf_algorithm(img_to_find)

    elif algorithm == 'ORB':
        print('ORB')
        keypoints_corner_1, descriptors_corner_1 = algorithms.orb_algorithm(img_corner_1)
        keypoints_corner_2, descriptors_corner_2 = algorithms.orb_algorithm(img_corner_2)
        keypoints_corner_3, descriptors_corner_3 = algorithms.orb_algorithm(img_corner_3)
        keypoints_corner_4, descriptors_corner_4 = algorithms.orb_algorithm(img_corner_4)

        keypoints_to_find, descriptors_to_find = algorithms.orb_algorithm(img_to_find)

    clusters = np.array([descriptors_corner_1])
    clusters1 = np.array([descriptors_corner_2])
    clusters2 = np.array([descriptors_corner_3])
    clusters3 = np.array([descriptors_corner_4])
    if matching_method == 'BRUTE-FORCE' and not algorithm == 'ORB':
        bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
        bf.add(clusters)
        bf.add(clusters1)
        bf.add(clusters2)
        bf.add(clusters3)
        bf.train()
        start = timer()
        matches = bf.knnMatch(descriptors_to_find, k=2)
        end = timer()
    elif matching_method == 'BRUTE-FORCE' and algorithm == 'ORB':
        print('BRUTE-FORCE + ORB')
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        bf.add(clusters)
        bf.add(clusters1)
        bf.add(clusters2)
        bf.add(clusters3)
        bf.train()
        start = timer()
        matches = bf.knnMatch(descriptors_to_find, k=2)
        end = timer()
    elif matching_method == 'FLANN' and not algorithm == 'ORB':
        FLAN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        flann.add(clusters)
        flann.add(clusters1)
        flann.add(clusters2)
        flann.add(clusters3)
        flann.train()
        start = timer()
        matches = flann.knnMatch(descriptors_to_find, k=2)
        end = timer()
    elif matching_method == 'FLANN' and algorithm == 'ORB':
        print('FLANN + ORB')
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        flann.add(clusters)
        flann.add(clusters1)
        flann.add(clusters2)
        flann.add(clusters3)
        flann.train()
        start = timer()
        matches = flann.knnMatch(descriptors_to_find, k=2)
        end = timer()

    time = end - start
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append(m)

    print(len(good))
    print(len(matches))
    print(len(good) / len(matches) * 100)

    with open(matching_method + "_matching_results_group.txt", 'a') as f:
        f.write(str(algorithm) + " <-> " + str(img_to_find_path) +"\n" + "\n")
        f.write("Ilosc dopasowan: " + str(
            len(good)) + ", procent dopasowan: " + str(
            (len(good)/len(descriptors_to_find))*100) + ", calkowity czas dopasowania: " + str("{:.10f}".format(time))+ "\n")
        f.write("\n" + "\n")

    good_1 = []
    good_2 = []
    good_3 = []
    good_4 = []

    for i in range(len(good)):
        if good[i].imgIdx == 0:
            good_1.append([good[i]])
        elif good[i].imgIdx == 1:
            good_2.append([good[i]])
        elif good[i].imgIdx == 2:
            good_3.append([good[i]])
        elif good[i].imgIdx == 3:
            good_4.append([good[i]])
        # print(good[i].imgIdx)

    if good_1:
        matched_img_1 = cv.drawMatchesKnn(img_to_find, keypoints_to_find, img_corner_1, keypoints_corner_1,
                                         good_1, None)
        cv.imshow("Matching Images_1", matched_img_1)
        # cv.imwrite(algorithm + "_" +  matching_method + "_1" + '.png', matched_img_1)
    if good_2:
        matched_img_2 = cv.drawMatchesKnn(img_to_find, keypoints_to_find, img_corner_2, keypoints_corner_2,
                                         good_2, None)
        cv.imshow("Matching Images_2", matched_img_2)
        # cv.imwrite(algorithm + "_" + matching_method + "_2" + '.png', matched_img_2)
    if good_3:
        matched_img_3 = cv.drawMatchesKnn(img_to_find, keypoints_to_find, img_corner_3, keypoints_corner_3,
                                         good_3, None)
        cv.imshow("Matching Images_3", matched_img_3)
        # cv.imwrite(algorithm + "_" +  matching_method + "_3" + '.png', matched_img_3)
    if good_4:
        matched_img_4 = cv.drawMatchesKnn(img_to_find, keypoints_to_find, img_corner_4, keypoints_corner_4,
                                         good_4, None)
        cv.imshow("Matching Images_4", matched_img_4)
        # cv.imwrite(algorithm + "_" +  matching_method + "_4" + '.png', matched_img_4)

    cv.waitKey(0)


group_of_descriptors('HOUSE/HD/Choice/TO_FIND_8.jpg', 'ORB', 'FLANN')
# test_img_matching('BOAT/img1.pgm', 'BOAT/img3.pgm', 'FLANN')
# test_img("LEUVEN")