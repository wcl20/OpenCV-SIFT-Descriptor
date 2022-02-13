import cv2
import os

def main():

    detector = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    image_path = os.path.join("images", "dog.jpg")
    image1 = cv2.imread(image_path)
    keypoints1, descriptors1 = detector.detectAndCompute(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), None)

    image_path = os.path.join("images", "dog.png")
    image2 = cv2.imread(image_path)
    keypoints2, descriptors2 = detector.detectAndCompute(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), None)


    matches = matcher.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    cv2.imshow("image", cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=2))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
