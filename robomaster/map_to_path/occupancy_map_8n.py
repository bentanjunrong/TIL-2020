from gridmap import OccupancyGridMap
import matplotlib.pyplot as plt
from a_star import a_star
from utils import plot_path
import cv2
import numpy

CNT = 0

def save_image(path, splash=False):
    global CNT

    start_x, start_y = path[0]
    goal_x, goal_y = path[-1]

    # plot path
    path_arr = numpy.array(path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], 'y')

    # plot start point
    plt.plot(start_x, start_y, 'ro')

    # plot goal point
    plt.plot(goal_x, goal_y, 'go')

    if splash:
        plt.savefig("output_frame_splash%d.png" % CNT)
    else:
        plt.savefig("output_frame%d.png" % CNT)

def process_path(gmap, start_node, end_node, splash=False):
    gmap.plot()

    path, path_px = a_star(start_node, end_node, gmap, movement='4N')

    if path:
        save_image(path, splash)
    else:
        print('Goal is not reachable', splash, CNT)

        # plot start and goal points over the map (in pixels)
        start_node_px = gmap.get_index_from_coordinates(start_node[0], start_node[1])
        goal_node_px = gmap.get_index_from_coordinates(end_node[0], end_node[1])

        plt.plot(start_node_px[0], start_node_px[1], 'ro')
        plt.plot(goal_node_px[0], goal_node_px[1], 'go')

        if splash:
            plt.savefig("output_frame_splash%d.png" % CNT)
        else:
            plt.savefig("output_frame%d.png" % CNT)

def plot_path(frame, start_node, end_node):
    global CNT

    # convert to png
    cv2.imwrite("input_frame%d.png" % CNT, frame)

    # plot path
    gmap_splash = OccupancyGridMap.from_png("input_frame%d.png" % CNT, 1, splash=True)
    gmap_normal = OccupancyGridMap.from_png("input_frame%d.png" % CNT, 1)

    process_path(gmap_splash, start_node, end_node, True)
    process_path(gmap_normal, start_node, end_node)

    CNT += 1

# testing funcs below

def start_video(is_flipped = False):
    global COL, ROW, GRID

    vid = '/home/rohit/Videos/3.avi'
    frame_cnt = 0
    cv2.namedWindow('fk tello', cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(vid)
    while not cap.isOpened():
        cap = cv2.VideoCapture(vid)
        print("loading video...")

        # esc to close stream
        if cv2.waitKey(16) == 27:
            break
    cnt = 0
    while True:
        if cnt > 2: break
        flag, frame = cap.read()
        if flag:
            plot_path(frame, (0,0), (0,1))
        cnt += 1

def start_image():
    plot_path(cv2.imread('maps/try8.png'), (850,100), (850,500))
    plot_path(cv2.imread('maps/try7.png'), (850,100), (850,500))
    return
