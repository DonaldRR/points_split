import pdb
import numpy as np
import copy
import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('--method', type=str, default="hard")
    args = parser.parse_args()
    return args

# Drawing methods
def draw_points(points):
    plt.scatter(points[:, 0], points[:, 1], s=2, c="red")

def draw_lines(lines):
    for line in lines:
        plt.plot(line[:, 0], line[:, 1], c="blue")

def generate_points(n_points=10):
    # Generate random test points

    points = np.random.rand(n_points, 2) - .5
    points = (points*10) + .5
    points = list(points.astype(int))
    points = [tuple(point) for point in points]
    if len(set(points)) < n_points:
        return generate_points()
    points = sorted(points, key=lambda x: x[0])
    points = np.array([list(point) for point in points], dtype=float)
    return points

def hard_split(points):
    # Split points into intervals across x-axis, and split points in each interval   

    y_min = min(points[:, 1])
    x_min = min(points[:, 0])
    x_max = max(points[:, 0])
    left = x_min - .5
    cur_x = None
    intervals = []
    i = 0
    lines = []
    slots = []
    while i < len(points):
        slot = []
        x = points[i][0]
        slot.append(points[i])
        j = i + 1
        while j < len(points) and x == points[j][0]:
            slot.append(points[j])
            j+=1
        i = j
        slot = sorted(slot, key=lambda x: x[1])
        slots.append(slot)

        if i < len(points):
            right = (slot[-1][0] + points[i][0]) / 2
        else:
            right = slot[-1][0] + .5
        intervals.append([left, right])
        lines.append([[left, y_min-.5], [right, y_min-.5]])
        
        if len(slot) > 1:
            separates = []
            k = 0
            while k < len(slot) - 1:
                separate = (slot[k][1] + slot[k+1][1]) / 2
                lines.append([[left, separate], [(x + right) / 2, separate]])
                separates.append(separate)
                # plt.scatter([right], [separate], color="red")
                k += 1

        # plt.scatter([left], [y_min - .5], color="red")
        # plt.scatter([right], [y_min - .5], color="red")

        left = right
    
    for i, interval in enumerate(intervals):
        if i < len(intervals) - 1:
            lines.append([[interval[0], y_min-.5], [interval[0], max(slots[i][-1][1], slots[i+1][-1][1])]])
        else:
            lines.append([[interval[0], y_min-.5], [interval[0], slots[i][-1][1]]])

    lines = np.array(lines)

    return lines

def intersect_lines(line1, line2):
    # find if two lines are parallel or not, if not return the intersected point

    a1 = line1[1, 1] - line1[0, 1]
    b1 = line1[0, 0] - line1[1, 0]
    c1 = line1[1, 0] * line1[0, 1] - line1[0, 0] * line1[1, 1]
    d1 = np.sqrt(a1**2+b1**2)
    a1, b1, c1 = a1/d1, b1/d1, c1/d1

    a2 = line2[1, 1] - line2[0, 1]
    b2 = line2[0, 0] - line2[1, 0]
    c2 = line2[1, 0] * line2[0, 1] - line2[0, 0] * line2[1, 1]
    d2 = np.sqrt(a2**2+b2**2)
    a2, b2, c2 = a2/d2, b2/d2, c2/d2

    if a1==a2 and b1==b2:
        return False, line1, line2
    x = (c1 - b1/b2*c2) / (b1/b2*a2 - a1) if b2 != 0 else (-c1 + b1*(a1/a2*c2-c1)/(a1/a2*b2-b1)) / a1
    y = (c1 - a1/a2*c2) / (a1/a2*b2 - b1) if a2 != 0 else (-c1 + a1*(b1/b2*c2-c1)/(b1/b2*a2-a1)) / b1
    pt = np.array([[x, y]], dtype=float)

    offsets1 = pt - line1
    if (offsets1[0] * offsets1[1]).sum() > 0:
        # extend line1
        dist1 = np.sqrt((offsets1**2).sum(1))
        if dist1.min() > 15:
            return False, line1, line2
        if dist1[0] > dist1[1]:
            line1[1] = pt[0]
        else:
            line1[0] = pt[0]
    
    offsets2 = pt - line2
    if (offsets2[0] * offsets2[1]).sum() > 0:
        # extend line2
        dist2 = np.sqrt((offsets2**2).sum(1))
        if dist2.min() > 15:
            return False, line1, line2
        if dist2[0] > dist2[1]:
            line2[1] = pt[0]
        else:
            line2[0] = pt[0]

    return True, line1, line2

def merge_lines(lines):
    # connect line segments to one connected component

    connected_graph = {}
    connected_component = lines[:1]
    candidates = lines[1:]

    i = 0
    while len(connected_component) < len(lines):
        candidates_to_pop = []
        for j, candidate in enumerate(candidates):
            ret, l1, l2 = intersect_lines(connected_component[i], candidate)
            if ret:
                connected_component[i] = l1
                connected_component.append(l2)
                candidates_to_pop.insert(0, j)
        for j in candidates_to_pop:
            candidates.pop(j)
        i += 1
    
    return connected_component

def tree_split(points):
    # recusively split points in two sets with line

    if len(points) < 2:
        return []

    # find axis to project points on
    pca = PCA(n_components=1)
    pca.fit(points)

    cos_theta = pca.components_[0][0]
    sin_theta = -pca.components_[0][1]
    transform = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta],
    ], dtype=float)
    inv_transform = np.array([
        [cos_theta, sin_theta],
        [-sin_theta, cos_theta]
    ], dtype=float)
    # projected value
    proj = np.matmul(transform, points.T).T[:, 0]
    # find the projected median
    mean = proj.mean()
    mids = (proj[:-1] + proj[1:]) / 2
    min_dist = np.abs(proj).max()
    median = 0
    for mid in mids:
        if mid not in proj:
            if np.abs(mid - mean) < min_dist:
                min_dist = np.abs(mid - mean)
                median = mid
    # separated points cluster
    points1 = points[proj < median]
    points2 = points[proj >= median]
    # print("set1:", points1)
    # print("set2:", points2)

    # find the separating line
    mid_point = np.matmul(inv_transform, np.array([[median], [0]], dtype=float))
    cos_theta90 = -sin_theta
    sin_theta90 = cos_theta
    transform90 = np.array([
        [cos_theta90, -sin_theta90],
        [sin_theta90, cos_theta90]
    ], dtype=float)
    inv_transform90 = np.array([
        [cos_theta90, sin_theta90],
        [-sin_theta90, cos_theta90]
    ], dtype=float)
    proj_90 = np.matmul(transform90, points.T - mid_point).T[:, 0]
    separate = np.array([
        [proj_90.min() - .5, 0],
        [proj_90.max() + .5, 0]
    ], dtype=float)
    separate = np.matmul(inv_transform90, separate.T).T + mid_point.T
    # print("line:", separate)
    
    return [separate] + tree_split(points1) + tree_split(points2)


def split(points, method="TREE_SPLIT"):

    if method == "HARD_SPLIT":
        return hard_split(points)
    elif method == "TREE_SPLIT":
        lines = tree_split(points)
        lines = merge_lines(lines)
        return lines

if __name__ == "__main__":

    args = parse_args()

    points = generate_points(args.n)
    draw_points(points)

    if args.method == "hard":
        lines = split(points, "HARD_SPLIT")
    elif args.method == "tree":
        lines = split(points, "TREE_SPLIT")
    
    draw_lines(lines)

    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()