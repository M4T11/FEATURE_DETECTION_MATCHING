import cv2 as cv

def brute_force_match(descriptors_1, descriptors_2, ratio=False, orb=False):
    if ratio == True:
        if orb:
            bf = cv.BFMatcher(cv.NORM_HAMMING)
        else:
            bf = cv.BFMatcher(cv.NORM_L2)
        matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append([m])
        return good, matches
    else:
        if orb:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        else:
            bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
        matches = bf.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key=lambda match: match.distance)
        return matches


def FLANN_match(descriptors_1, descriptors_2, orb=False):
    if orb:
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
    else:
        FLAN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLAN_INDEX_KDTREE, trees=5)

    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append([m])

    return good_matches, matches