import utils
import cv2
import time

# initialize the camera
cap = cv2.VideoCapture(0)
face_cropped = None

while True:
    # with FPS
    start_time = time.time()

    # get the frame
    _, frame = cap.read()

    # get the bounding box
    bbs = utils.get_all_face_bbs_opencv(frame)
    bb = utils.get_biggest_bb(bbs)
    # draw the bounding box on the frame
    if bb is not None:
        frame = utils.draw_bbs_on_img(frame, bbs, (0, 255, 0))
        face_cropped = utils.crop_face(frame, bb)

    # get frame height
    height = frame.shape[0]

    #resize face_cropped to height
    if face_cropped is not None:
        face_cropped = cv2.resize(face_cropped, (height, height))

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
