from vs.constants import VS
from random import randint
import os

class Cluster:
    maps_received = 0
    has_clusterized = False
    unified_map = {}
    unified_victims_map = {}
    filtered_map = {}
    final_centroids = {0: set(),
                       1: set(),
                       2: set(),
                       3: set()}
    clusters = {
        0: {},
        1: {},
        2: {},
        3: {}
    }

    cluster_index = -1
    get_cluster_index = -1
    
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
                    lowest_squared_distance = (centroids[0][0]-coordinate[0])**2 + (centroids[0][1]-coordinate[1])**2
                    index = 0

                    for j in range(1, len(centroids)):
                        squared_distance = (centroids[j][0]-coordinate[0])**2 + (centroids[j][1]-coordinate[1])**2

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
            cls.__clear_data()
    
    @classmethod
    def __complement_cluster_data(cls):
        for key in cls.final_centroids.keys():
            data = {}
            for coordinate in cls.final_centroids[key]:
                cell_data = cls.unified_map[coordinate]
                data.update({coordinate:cell_data})
            cls.final_centroids.update({key:data})

    @classmethod
    def __get_clusters(cls):
        cls.cluster_index += 1
        return cls.final_centroids[cls.cluster_index]
    
    @classmethod
    def get_cluster(cls):
        cls.get_cluster_index += 1
        return cls.clusters[cls.cluster_index]
    
    @classmethod
    def __filter_map(cls):
        for key in cls.unified_map.keys():
            if cls.unified_map[key][1] != VS.NO_VICTIM:
                cls.filtered_map.update({key:cls.unified_map[key]})

    @classmethod
    def deliver_data(cls, map, victims):
        cls.unified_map.update(map)
        cls.unified_victims_map.update(victims)
        
        for index in range(0, 4):
            if not cls.clusters[index]:
                cls.clusters[index].update(victims)
                break 

    @classmethod
    def __write_data(cls, victim_id, x, y, severity, cluster_id):
        with open("clusters/cluster" + str(cls.cluster_index) + ".txt", "a") as file:
            file.write(
                str(victim_id) + "," +
                str(x) + "," +
                str(y) + "," +
                str(severity) + "," +
                str(cluster_id) + "\n"
            )
                    
    @classmethod
    def __clear_data(cls):
        files = os.listdir("clusters")
        for file in files:
            os.remove("clusters/" + file)     
            
    @classmethod
    def transfer_data(cls):
        cluster = cls.__get_clusters()
  
        for coordinate in cluster.keys():
            victim_id = cluster[coordinate][1]

            if victim_id in cls.unified_victims_map.keys():
                x = coordinate[0]
                y = coordinate[1]
                severity = cls.unified_victims_map[victim_id][1][1]
                cluster_id = cls.cluster_index 

                cls.__write_data(victim_id, x, y, severity, cluster_id)
