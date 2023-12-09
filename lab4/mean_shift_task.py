import matplotlib.pyplot as plt
import numpy as np
import random

STOP_THRESHOLD = 1e-4
CLUSTER_THRESHOLD = 1e-1


def gaussian_cluster(cx: float = 0.0, cy: float = 0.0, sigma_x: float = 0.1, sigma_y: float = 0.1, n_points: int = 1024):
    """
    Двумерный кластер точек, распределённых нормально с центром в
    точке с координатами cx, cy и разбросом sigma_x, sigma_y.
    """
    return np.hstack((np.random.normal(cx, sigma_x, n_points).reshape((n_points, 1)),
                      np.random.normal(cy, sigma_y, n_points).reshape((n_points, 1))))


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def gaussian_kernel(distance, bandwidth):
    return (1 / (bandwidth * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((distance / bandwidth)) ** 2)


class MeanShift(object):
    def __init__(self, kernel=gaussian_kernel):
        self.kernel = kernel

    def fit(self, points, kernel_bandwidth):

        shift_points = np.array(points)
        shifting = [True] * points.shape[0]

        while True:
            max_dist = 0
            for i in range(0, len(shift_points)):
                if not shifting[i]:
                    continue
                p_shift_init = shift_points[i].copy()
                shift_points[i] = self._shift_point(
                    shift_points[i], points, kernel_bandwidth)
                dist = distance(shift_points[i], p_shift_init)
                max_dist = max(max_dist, dist)
                shifting[i] = dist > STOP_THRESHOLD

            if (max_dist < STOP_THRESHOLD):
                break
        cluster_ids = self._cluster_points(shift_points.tolist())
        return shift_points, cluster_ids

    def _shift_point(self, point, points, kernel_bandwidth):
        points = np.array(points)
        dist = np.linalg.norm(points - point, axis=1)
        weight = self.kernel(dist, kernel_bandwidth)
        scale = np.sum(weight)
        shift = (weight @ points) / scale
        return shift

    def _cluster_points(self, points):
        cluster_ids = []
        cluster_idx = 0
        cluster_centers = []

        for i, point in enumerate(points):
            if (len(cluster_ids) == 0):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
            else:
                for center in cluster_centers:
                    dist = distance(point, center)
                    if (dist < CLUSTER_THRESHOLD):
                        cluster_ids.append(cluster_centers.index(center))
                if (len(cluster_ids) < i + 1):
                    cluster_ids.append(cluster_idx)
                    cluster_centers.append(point)
                    cluster_idx += 1
        return cluster_ids


def colors(n):
    ret = []
    for i in range(n):
        ret.append((random.uniform(0, 1), random.uniform(
            0, 1), random.uniform(0, 1)))
    return ret


def debuf_color(color):
    debuf_color = tuple(c * 0.7 for c in color)
    return debuf_color


def separated_clusters():
    """
    Пример с пятью разрозненными распределениями точек на плоскости.
    """
    n = 100
    clusters_data = np.vstack((gaussian_cluster(cx=0.5, n_points=n),
                               gaussian_cluster(cx=1.0, n_points=n),
                               gaussian_cluster(cx=1.5, n_points=n),
                               gaussian_cluster(cx=2.0, n_points=n),
                               gaussian_cluster(cx=2.5, n_points=n)))
    mean_shifter = MeanShift()

    shift_points, mean_shift_result = mean_shifter.fit(
        clusters_data, kernel_bandwidth=0.15)

    np.set_printoptions(precision=3)
    color = colors(np.unique(mean_shift_result).size)
    for i in range(len(mean_shift_result)):
        plt.scatter(clusters_data[i, 0], clusters_data[i,
                    1], color=color[mean_shift_result[i]])
        plt.scatter(shift_points[i][0], shift_points[i][1], color=debuf_color(
            color[mean_shift_result[i]]), marker='x')
    plt.show()


def merged_clusters():
    """
    Пример с кластеризацией пятна.
    """
    clusters_data = gaussian_cluster(n_points=100)
    mean_shifter = MeanShift()
    shift_points, mean_shift_result = mean_shifter.fit(
        clusters_data, kernel_bandwidth=0.15)

    np.set_printoptions(precision=3)
    color = colors(np.unique(mean_shift_result).size)
    for i in range(len(mean_shift_result)):
        plt.scatter(clusters_data[i, 0], clusters_data[i,
                    1], color=color[mean_shift_result[i]])
        plt.scatter(shift_points[i][0], shift_points[i][1], color=debuf_color(
            color[mean_shift_result[i]]), marker='x')
    plt.show()


if __name__ == "__main__":
    """
    Если вы это читаете, то должны поставить автомат!!! Вызов функций "merged_clusters" и "separated_clusters".
    """
    merged_clusters()
    separated_clusters()
