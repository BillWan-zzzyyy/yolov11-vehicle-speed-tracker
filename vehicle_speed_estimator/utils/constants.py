import numpy as np

# width: 3.66 meters height: 9.14 meters

#VeronaRd
IMAGE_POINTS = [(591, 260), (1167, 213), (2517, 668), (1047, 1117)]
WORLD_POINTS = [(0, 0), (np.float64(16.89971446299426), 0), (np.float64(16.89971446299426), np.float64(50.455152595905915)), (0, np.float64(50.455152595905915))]

#default for test
# IMAGE_POINTS = [(800, 410), (1125, 410), (1920, 850), (0, 850)]
# WORLD_POINTS = [(0, 0), (32, 0), (32, 140), (0, 140)]


# IMAGE_POINTS = [(8, 957), (671, 480), (1297, 486), (1919, 963)]
# WORLD_POINTS = [(0, 0), (np.float64(16.794067808947265), 0), (np.float64(16.794067808947265), np.float64(39.29378917984479)), (0, np.float64(39.29378917984479))]

# #test3
# IMAGE_POINTS = [(679, 445), (1227, 451), (1889, 793), (71, 772)]
# WORLD_POINTS = [(0, 0), (np.float64(9.783343758759592), 0), (np.float64(9.783343758759592), np.float64(39.05249799948781)), (0, np.float64(39.05249799948781))]
# POLYGON = np.array(IMAGE_POINTS)

# #test2
# IMAGE_POINTS = [(242, 133), (1141, 151), (1161, 692), (212, 682)]
# WORLD_POINTS = [(0, 0), (np.float64(17.773912512228204), 0), (np.float64(17.773912512228204), np.float64(12.034769005084781)), (0, np.float64(12.034769005084781))]
POLYGON = np.array(IMAGE_POINTS)