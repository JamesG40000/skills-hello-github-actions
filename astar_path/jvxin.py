import cv2
import numpy as np
import time
from heapq import heappush, heappop


def parse_maze_image(img_path):

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("无法读取图像，请检查路径或文件格式。")

    hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w, _ = hsv_img.shape

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 100, 70])
    upper_green = np.array([80, 255, 255])

    red_mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    green_mask = cv2.inRange(hsv_img, lower_green, upper_green)

    maze = np.zeros((h, w), dtype=bool)

    start = None
    goal = None

    pass_threshold = 200

    for y in range(h):
        for x in range(w):
            if red_mask[y, x] > 0:

                if start is None:
                    start = (y, x)
                maze[y, x] = True
            elif green_mask[y, x] > 0:
                if goal is None:
                    goal = (y, x)
                maze[y, x] = True
            else:
                b, g, r = img_bgr[y, x]
                if b > pass_threshold and g > pass_threshold and r > pass_threshold:
                    maze[y, x] = True
                else:
                    maze[y, x] = False

    if start is None or goal is None:
        raise ValueError("没有检测到起点或终点，请检查红/绿色阈值或图像内容。")

    return maze, start, goal, img_bgr


def get_neighbors(maze, r, c):
    neighbors = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1]:
            if maze[nr, nc]:
                neighbors.append((nr, nc))
    return neighbors


def bfs(maze, start, goal):
    from collections import deque
    visited = set()
    queue = deque()
    queue.append((start, [start]))
    visited.add(start)

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == goal:
            return path

        for nxt in get_neighbors(maze, r, c):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return None


def dijkstra(maze, start, goal):

    dist = {start: 0}
    visited = set()
    pq = []
    heappush(pq, (0, start, [start]))

    while pq:
        current_dist, current_node, path = heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal:
            return path

        for nxt in get_neighbors(maze, *current_node):
            if nxt not in visited:
                old_dist = dist.get(nxt, float('inf'))
                new_dist = current_dist + 1
                if new_dist < old_dist:
                    dist[nxt] = new_dist
                    heappush(pq, (new_dist, nxt, path + [nxt]))
    return None


def a_star(maze, start, goal):
    def heuristic(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    dist = {start: 0}
    visited = set()
    pq = []  # (f, 当前节点, 路径)，其中 f = g + h
    heappush(pq, (heuristic(start, goal), start, [start]))

    while pq:
        f, current_node, path = heappop(pq)
        if current_node in visited:
            continue
        visited.add(current_node)

        if current_node == goal:
            return path

        for nxt in get_neighbors(maze, *current_node):
            if nxt not in visited:
                tentative_g = dist[current_node] + 1
                if tentative_g < dist.get(nxt, float('inf')):
                    dist[nxt] = tentative_g
                    f_nxt = tentative_g + heuristic(nxt, goal)
                    heappush(pq, (f_nxt, nxt, path + [nxt]))
    return None


def draw_path_on_image(img, path, color=(0, 0, 255), thickness=1):

    out_img = img.copy()
    for i in range(len(path) - 1):
        (r1, c1) = path[i]
        (r2, c2) = path[i+1]
        cv2.line(out_img, (c1, r1), (c2, r2), color, thickness)
    return out_img


def main():
    maze_path = "square.png"
    maze, start, goal, original_img = parse_maze_image(maze_path)

    t0 = time.process_time()
    path_bfs = bfs(maze, start, goal)
    t1 = time.process_time()
    bfs_time = t1 - t0

    t0 = time.process_time()
    path_dijk = dijkstra(maze, start, goal)
    t1 = time.process_time()
    dijk_time = t1 - t0

    t0 = time.process_time()
    path_astar = a_star(maze, start, goal)
    t1 = time.process_time()
    astar_time = t1 - t0

    print("BFS Time:       {:.6f}s, Path Length: {}, Color: Blue".format(bfs_time, len(path_bfs) if path_bfs else None))
    print("Dijkstra Time:  {:.6f}s, Path Length: {}, Color: Green".format(dijk_time,
                                                                          len(path_dijk) if path_dijk else None))
    print("A* Time:        {:.6f}s, Path Length: {}, Color: Red".format(astar_time,
                                                                        len(path_astar) if path_astar else None))

    if path_bfs:
        bfs_img = draw_path_on_image(original_img, path_bfs, color=(255, 0, 0), thickness=1)  # 蓝色
        cv2.imwrite("bfs_path.png", bfs_img)
    if path_dijk:
        dijk_img = draw_path_on_image(original_img, path_dijk, color=(0, 255, 0), thickness=1)  # 绿色
        cv2.imwrite("dijkstra_path.png", dijk_img)
    if path_astar:
        astar_img = draw_path_on_image(original_img, path_astar, color=(0, 0, 255), thickness=1)  # 红色
        cv2.imwrite("astar_path.png", astar_img)

    # 如果需要将三条路径同时绘制在同一张图上：
    # combined_img = original_img.copy()
    # if path_bfs:
    #     combined_img = draw_path_on_image(combined_img, path_bfs,  color=(255,   0,   0), thickness=1)
    # if path_dijk:
    #     combined_img = draw_path_on_image(combined_img, path_dijk, color=(  0, 255,   0), thickness=1)
    # if path_astar:
    #     combined_img = draw_path_on_image(combined_img, path_astar,color=(  0,   0, 255), thickness=1)
    # cv2.imwrite("combined_paths.png", combined_img)


if __name__ == "__main__":
    main()
