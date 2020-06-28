import png
import cv2
import asyncio

THRESHOLD = 80                              # black threshold
LENGTH = 9                                  # arena length
HEIGHT = 3                                  # arena height
GRID = [[-1]*LENGTH for i in range(HEIGHT)] # occupancy grid
ROW, COL = 0, 0                             # current grid position

async def take_pic(frame, last=False):
    global COL, ROW, GRID

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    res = 0
    for row in img:
        res += sum(row)
    res /= sum([len(row) for row in img])
    GRID[ROW % HEIGHT][COL % LENGTH] = 1 if res > THRESHOLD else 0
    COL += 1
    # ROW = ROW + 1 if not COL % LENGTH else ROW # auto ROW change
    ROW = ROW + 1 if last else ROW

def start_video(is_flipped = False):
    global COL, ROW, GRID

    vid = 'dronesucks.mp4'
    # vid = 'PitchVideo.mp4'
    # vid = 'spectrum1.mp4'
    # vid = 'spectrum2.mp4'
    frame_cnt = 0
    loop = asyncio.get_event_loop()
    cv2.namedWindow('fk tello', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(vid)
    while not cap.isOpened():
        cap = cv2.VideoCapture(vid)
        print("loading video...")

        # esc to close stream
        if cv2.waitKey(16) == 27:
            break

    while True:
        flag, frame = cap.read()

        # esc to close stream
        if cv2.waitKey(16) == 27:
            break

        if cv2.waitKey(16) == 112:
            # p key to take pic
            print('taking photo...')
            val = loop.run_until_complete(take_pic(frame)) # 0 (pitch black) and 255 (bright white)
            print(GRID)
            print('photo taken and analysed successfully.')

        if cv2.waitKey(16) == 108:
            # l key to take last pic
            print('taking last photo...')
            val = loop.run_until_complete(take_pic(frame, True)) # 0 (pitch black) and 255 (bright white)
            print(GRID)
            print('photo taken and analysed successfully.')

        if cv2.waitKey(16) == 114:
            # r key to revert last, if exists
            print('reverting last photo...')
            GRID[ROW % HEIGHT][COL % LENGTH] = -1
            ROW = ROW - 1 if ROW and not COL % LENGTH else ROW
            COL = COL - 1 if COL else COL
            print('last photo reverted.')

        # on successful read of frame
        if flag:
            if is_flipped:
                cv2.imshow('fk tello', cv2.flip(frame, 0))
            else:
                cv2.imshow('fk tello', frame)
            frame_cnt = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(frame_cnt) + " frames")

        # retry loading last frame on fail; only for non livestreams
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt - 1)
            print("frame is not ready")
            cv2.waitKey(1000)

        # since its not a livestream, need break condition
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # stop if all frames exhausted
            break

    loop.close()


start_video()
