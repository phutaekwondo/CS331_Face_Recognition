from mtcnn import MTCNN
import tensorflow as tf
import cv2

# tf.debugging.set_log_device_placement(True)
mtcnn_detector = None
opencv_face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def mtcnn_init():
    """
    Initialize MTCNN
    :return: MTCNN object
    """
    print('Initializing MTCNN...')
    global mtcnn_detector 
    mtcnn_detector = MTCNN() 

def get_biggest_bb(bbs):
    """
    Get the biggest bounding box
    :param bbs: bounding boxes
    :return: biggest bounding box
    """
    if bbs is None: return None
    biggest_bb = None
    biggest_bb_area = 0
    for bb in bbs:
        area = (bb[2] - bb[0]) * (bb[3] - bb[1])
        if area > biggest_bb_area:
            biggest_bb_area = area
            biggest_bb = bb
    return biggest_bb

def get_all_face_bbs_opencv(img):
    """
    Get the bounding box of the face in the image
    :param img: image
    :return: bounding box of the face
    """
    # detect faces in the image
    faces = opencv_face_detector.detectMultiScale(img, 1.3, 5)

    if len (faces) <= 0: return None
    # extract the bounding box from the first face
    bbs = []
    for face in faces:
        x1, y1, width, height = face
        x2, y2 = x1 + width, y1 + height
        bbs.append((x1, y1, x2, y2))
    return bbs

def get_face_bb_opencv(img):
    """
    Get the bounding box of the face in the image
    :param img: image
    :return: bounding box of the face
    """
    # detect faces in the image
    face_bbs = get_all_face_bbs_opencv(img)
    return get_biggest_bb(face_bbs)


def get_face_bb_mtcnn(img):
    """
    Get the bounding box of the face in the image
    :param img: image
    :return: bounding box of the face
    """

    if mtcnn_detector is None:
        mtcnn_init()

    # detect faces in the image
    faces = mtcnn_detector.detect_faces(img)

    if len (faces) <= 0: return None
    # extract the bounding box from the first face
    x1, y1, width, height = faces[0]['box']
    x2, y2 = x1 + width, y1 + height
    return x1, y1, x2, y2

def crop_face(img, bb):
    """
    Crop the face from the image
    :param img: image
    :param bb: bounding box
    :return: cropped face
    """
    return img[bb[1]:bb[3], bb[0]:bb[2]]

def draw_bb_on_img(img, bb, color):
    """
    Draw bounding box on the image
    :param img: image
    :param bb: bounding box
    :return: image with bounding box
    """
    # draw the bounding box
    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
    return img


