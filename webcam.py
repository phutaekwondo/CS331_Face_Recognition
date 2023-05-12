import utils
import cv2
import time
from arcface import ArcFaceModel

# initialize the camera
cap = cv2.VideoCapture(0)
face_cropped = None

arcface = ArcFaceModel()

face_cuong = cv2.imread('img/cuong.jpg')
face_phu = cv2.imread('img/phu.jpg')

arcface.register_face('cuong', face_cuong, True)
arcface.register_face('phu', face_phu)

while True:
    # with FPS
    start_time = time.time()
    # ret, frame = cap.read()
    # if not ret: break
    # frame = cv2.flip(frame, 1)
    # frame = cv2.resize(frame, (640, 480))
    # bb = utils.get_face_bb(frame)
    # if bb is not None:
    #     frame = utils.draw_bb_on_img(frame, bb, (0, 255, 0))
    # cv2.putText(frame, "FPS: {:.2f}".format(1.0 / (time.time() - start_time)), (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # cv2.imshow("Frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'): break

    # get the frame
    _, frame = cap.read()

    # get the bounding box
    bb = utils.get_face_bb_opencv(frame)
    # draw the bounding box on the frame
    if bb is not None:
        frame = utils.draw_bb_on_img(frame, bb, (0, 255, 0))
        face_cropped = utils.crop_face(frame, bb)

    # get frame height
    height = frame.shape[0]

    #resize face_cropped to height
    if face_cropped is not None:

        # get the name of the person
        name = arcface.recognize_face(face_cropped)

        face_cropped = cv2.resize(face_cropped, (height, height))

        # draw the name of the person
        if name is not None and bb is not None:
            cv2.putText(frame, name, (bb[0], bb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    #concat frame and face_cropped
    frame = cv2.hconcat([frame, face_cropped])

    #show the FPS
    cv2.putText(frame, "FPS: {:.2f}".format(1.0 / (time.time() - start_time)), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # show the frame
    cv2.imshow("frame", frame)
    # wait for key press
    key = cv2.waitKey(1)
    # if the key pressed is `q` then break from the loop
    if key == ord("q"):
        break

# release the camera
cap.release()
# close all the frames
cv2.destroyAllWindows()
