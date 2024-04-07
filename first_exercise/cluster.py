from vs.constants import VS
from random import randint

class Cluster:
    maps_received = 0
    has_clusterized = False
    unified_map = {}
    filtered_map = {}
    final_centroids = {0: set(),
                       1: set(),
                       2: set(),
                       3: set()}
    cluster_index = -1
    
    @classmethod
    def k_means(cls):
        if not cls.has_clusterized:
            cls.has_clusterized = True
            cls.__filter_map()

            k = 4
            coordinates = list(cls.filtered_map.keys())
            centroids = []

            for i in range(k):
                index = randint(0, len(coordinates) - 1) 
                centroids.append(coordinates[index])
                coordinates.pop(index)

            max_iterations = 1000
            i = 0
            centroid_has_changed = True

            while i <= max_iterations and centroid_has_changed:
                centroid_has_changed = False

                for coordinate in cls.filtered_map.keys():
                    lowest_squared_distance = (centroids[0][0]-coordinate[0])*(centroids[0][0]-coordinate[0]) + (centroids[0][1]-coordinate[1])*(centroids[0][1]-coordinate[1])
                    index = 0

                    for j in range(1, len(centroids)):
                        squared_distance = (centroids[j][0]-coordinate[0])*(centroids[j][0]-coordinate[0]) + (centroids[j][1]-coordinate[1])*(centroids[j][1]-coordinate[1])

                        if squared_distance < lowest_squared_distance:
                            lowest_squared_distance = squared_distance
                            index = j

                    if coordinate in cls.final_centroids[0]:
                        cls.final_centroids[0].discard(coordinate)
                    elif coordinate in cls.final_centroids[1]:
                        cls.final_centroids[1].discard(coordinate)
                    elif coordinate in cls.final_centroids[2]:
                        cls.final_centroids[2].discard(coordinate)
                    elif coordinate in cls.final_centroids[3]:
                        cls.final_centroids[3].discard(coordinate)

                    centroids_set = cls.final_centroids[index]
                    centroids_set.add(coordinate)
                    cls.final_centroids.update({index:centroids_set})

                for j in range(k):
                    coordinates_list = cls.final_centroids[j]
                    x = 0
                    y = 0

                    for coordinate in coordinates_list:
                        x += coordinate[0]
                        y += coordinate[1]
                    
                    x = int(x / len(coordinates_list))
                    y = int(y / len(coordinates_list))

                    if (x, y) != centroids[j]:
                        centroid_has_changed = True
                    
                    centroids[j] = (x, y)

                i += 1

            cls.__complement_cluster_data()
    
    @classmethod
    def __complement_cluster_data(cls):
        for key in cls.final_centroids.keys():
            data = {}
            for coordinate in cls.final_centroids[key]:
                cell_data = cls.unified_map[coordinate]
                data.update({coordinate:cell_data})
            cls.final_centroids.update({key:data})

    @classmethod
    def get_clusters(cls):
        cls.cluster_index += 1
        return cls.final_centroids[cls.cluster_index]
    
    @classmethod
    def __filter_map(cls):
        for key in cls.unified_map.keys():
            if cls.unified_map[key][1] != VS.NO_VICTIM:
                cls.filtered_map.update({key:cls.unified_map[key]})

    @classmethod
    def deliver_data(cls, map, victims):
        cls.unified_map.update(map)

