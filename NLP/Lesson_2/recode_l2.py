import re

coordination_source = '''
{name:'兰州', geoCoord:[103.73, 36.03]},
{name:'嘉峪关', geoCoord:[98.17, 39.47]},
{name:'西宁', geoCoord:[101.74, 36.56]},
{name:'成都', geoCoord:[104.06, 30.67]},
{name:'石家庄', geoCoord:[114.48, 38.03]},
{name:'拉萨', geoCoord:[102.73, 25.04]},
{name:'贵阳', geoCoord:[106.71, 26.57]},
{name:'武汉', geoCoord:[114.31, 30.52]},
{name:'郑州', geoCoord:[113.65, 34.76]},
{name:'济南', geoCoord:[117, 36.65]},
{name:'南京', geoCoord:[118.78, 32.04]},
{name:'合肥', geoCoord:[117.27, 31.86]},
{name:'杭州', geoCoord:[120.19, 30.26]},
{name:'南昌', geoCoord:[115.89, 28.68]},
{name:'福州', geoCoord:[119.3, 26.08]},
{name:'广州', geoCoord:[113.23, 23.16]},
{name:'长沙', geoCoord:[113, 28.21]},
//{name:'海口', geoCoord:[110.35, 20.02]},
{name:'沈阳', geoCoord:[123.38, 41.8]},
{name:'长春', geoCoord:[125.35, 43.88]},
{name:'哈尔滨', geoCoord:[126.63, 45.75]},
{name:'太原', geoCoord:[112.53, 37.87]},
{name:'西安', geoCoord:[108.95, 34.27]},
//{name:'台湾', geoCoord:[121.30, 25.03]},
{name:'北京', geoCoord:[116.46, 39.92]},
{name:'上海', geoCoord:[121.48, 31.22]},
{name:'重庆', geoCoord:[106.54, 29.59]},
{name:'天津', geoCoord:[117.2, 39.13]},
{name:'呼和浩特', geoCoord:[111.65, 40.82]},
{name:'南宁', geoCoord:[108.33, 22.84]},
//{name:'西藏', geoCoord:[91.11, 29.97]},
{name:'银川', geoCoord:[106.27, 38.47]},
{name:'乌鲁木齐', geoCoord:[87.68, 43.77]},
{name:'香港', geoCoord:[114.17, 22.28]},
{name:'澳门', geoCoord:[113.54, 22.19]}
'''

def get_city_info(city_coordination):
    city_location = {}
    for line in city_coordination.split('\n'):
        if line.startswith('//'):
            continue
        if line.strip() == '':
            continue

        city = re.findall("name:'(\w+)'",line)[0]
        x_y = re.findall("Coord:\[(\d+.\d+),\s(\d+.\d+)]", line)[0]
        x_y = tuple(map(float, x_y))
        city_location[city] = x_y
    return city_location

print(get_city_info(coordination_source)['兰州'])

city_info = get_city_info(coordination_source)

import math

def geo_distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    # >>> origin = (48.1372, 11.5756)  # Munich
    # >>> destination = (52.5186, 13.4083)  # Berlin
    # >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def get_city_distance(city1, city2):
    return geo_distance(city_info[city1], city_info[city2])

print(get_city_distance('上海','杭州'))

import networkx as nx
import matplotlib.pyplot as plt

city_graph = nx.Graph()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
city_graph.add_nodes_from(list(city_info.keys()))
nx.draw(city_graph, city_info, with_labels=True, node_size=10)
# plt.show()

threshold = 700

from collections import defaultdict

def build_connection(city_info):
    cities_connection = defaultdict(list)
    cities = list(city_info.keys())
    for c1 in cities:
        for c2 in cities:
            if c1 == c2:
                continue
            if get_city_distance(c1, c2) < threshold:
                cities_connection[c1].append(c2)
    return cities_connection

cities_connection = build_connection(city_info)

# BFS 1 version

def search_1(graph, start, destination):
    pathes = [[start]]
    visited = set()

    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]

        if froniter in visited : continue

        successors = graph[froniter]

        for city in successors:
            if city in path: continue

            new_path = path + [city]

            pathes.append(new_path)

            if city == destination:
                return new_path
        visited.add(froniter)

# print(search_1(cities_connection,'上海','香港'))

def search_2(graph, start, destination, search):
    pathes = [[start]]
    visited = set()
    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]
        if froniter in visited: continue

        if froniter == destination:
            return path

        successors = graph[froniter]

        for city in successors:
            if city in path : continue

            new_path = path + [city]

            pathes.append(new_path)

        pathes = search(pathes)
        visited.add(froniter)

def sort_by_distance(pathes):
    def get_distance_of_path(path):
        distance = 0
        for i, _ in enumerate(path[:-1]):
            distance += get_city_distance(path[i], path[i+1])
        return distance
    return sorted(pathes, key=get_distance_of_path)
