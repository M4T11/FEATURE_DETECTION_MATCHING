import cv2
import pickle

class feature_detection_ROI(object):
    def __init__(self, path_to_video='fedex.mp4'):
        self.capture = cv2.VideoCapture(path_to_video)
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False
        self.update()

    def choose_algorithm(self, img):
        self.algorithms = {
            "SIFT": self.sift_on_ROI,
            "SURF": self.surf_on_ROI,
            "FAST_SURF": self.fast_surf_on_ROI,
            "ORB": self.orb_on_ROI,
        }
        for c, desc in self.algorithms.items():
            print(f"{c}. {desc}")

        choice = input("Select your algorithm: ")
        while choice not in self.algorithms:
            choice = input(f"Choose one of: {', '.join(self.algorithms)}: ")

        self.algorithms[choice](img)

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                cv2.imshow('image', self.frame)
                key = cv2.waitKey(2)

                if key == ord('c'):
                    self.clone = self.frame.copy()
                    cv2.namedWindow('image')
                    cv2.setMouseCallback('image', self.get_coordinates)
                    while True:
                        key = cv2.waitKey(2)
                        cv2.imshow('image', self.clone)

                        if key == ord('c'):
                            self.crop_ROI()
                            cv2.imshow('cropped image', self.cropped_image)
                            self.choose_algorithm(self.cropped_image)

                        if key == ord('r'):
                            break
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(1)
            else:
                pass

    def get_coordinates(self, event, x, y, flags, parameters):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False
            self.selected_ROI = True
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.frame.copy()
            self.selected_ROI = False

    def crop_ROI(self):
        if self.selected_ROI:
            self.cropped_image = self.frame.copy()

            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]

            self.cropped_image = self.cropped_image[y1:y2, x1:x2]

        else:
            print('Before crop select ROI')

    def sift_on_ROI(self, img):
        print('SIFT')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        index = []
        description = input("Enter description for object: ")
        for p in keypoints:
            temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
            index.append(temp)
        print(index)
        print(descriptors)
        output_image = cv2.drawKeypoints(gray, keypoints, 0, (255, 0, 0),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('output_image', output_image)
        cv2.imwrite('SIFT_ROI_images/' + str(description) + '.png', output_image)
        with open('SIFT.pickle', mode='ab+') as f:
            pickle.dump((index, descriptors, description), f)

    def surf_on_ROI(self, img):
        print('SURF')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints, descriptors = surf.detectAndCompute(gray, None)
        index = []
        description = input("Enter description for object: ")
        for p in keypoints:
            temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
            index.append(temp)
        print(index)
        print(descriptors)
        output_image = cv2.drawKeypoints(gray, keypoints, 0, (255, 0, 0),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('output_image', output_image)
        cv2.imwrite('SURF_ROI_images/' + str(description) + '.png', output_image)
        with open('SURF.pickle', mode='ab+') as f:
            pickle.dump((index, descriptors, description), f)

    def fast_surf_on_ROI(self, img):
        print('FAST-SURF')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fast = cv2.FastFeatureDetector_create()
        fast.setNonmaxSuppression(False)
        surf = cv2.xfeatures2d.SURF_create()
        keypoints = fast.detect(gray, None)
        _, descriptors = surf.compute(gray, keypoints)
        index = []
        description = input("Enter description for object: ")
        for p in keypoints:
            temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
            index.append(temp)
        print(index)
        print(descriptors)
        output_image = cv2.drawKeypoints(gray, keypoints, 0, (255, 0, 0),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('output_image', output_image)
        cv2.imwrite('FAST_SURF_ROI_images/' + str(description) + '.png', output_image)
        with open('FAST_SURF.pickle', mode='ab+') as f:
            pickle.dump((index, descriptors, description), f)

    def orb_on_ROI(self, img):
        print('ORB')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(1000, 1.2)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        print(keypoints)
        print(descriptors)
        index = []
        description = input("Enter description for object: ")
        for p in keypoints:
            temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
            index.append(temp)
        print(index)
        print(descriptors)
        output_image = cv2.drawKeypoints(gray, keypoints, 0, (255, 0, 0),
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('output_image', output_image)
        cv2.imwrite('ORB_ROI_images/' + str(description) + '.png', output_image)
        with open('ORB.pickle', mode='ab+') as f:
            pickle.dump((index, descriptors, description), f)

if __name__ == '__main__':
    feature_detection_ROI = feature_detection_ROI()