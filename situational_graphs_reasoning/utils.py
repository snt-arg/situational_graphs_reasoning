import numpy as np
import time
import math
import transforms3d.euler as eul
import itertools

def plane_4_params_to_6_params(plane):
    normal = plane[:3]
    distance = plane[3]
    point = -distance*np.array(normal)
    return(np.concatenate((point, normal)))


def plane_6_params_to_4_params(point_and_normal):
    point = np.array(point_and_normal[:3])
    normal = np.array(point_and_normal[3:6])
    ### OLD
    # distance = - np.dot(point, normal)
    ### NEW
    closest_point = closest_point_on_line(point, np.array([0,0,0]), normal)
    distance = -1 * np.sign(np.dot(closest_point, normal)) * np.linalg.norm(closest_point)
    ### end
    return(np.concatenate((normal, [distance])))

def closest_point_on_line(point, line_origin, line_normal):
    line_normal_unit = line_normal / np.linalg.norm(line_normal)
    vector_to_point = point - line_origin
    distance_along_normal = np.dot(vector_to_point, line_normal_unit)
    closest_point = line_origin + distance_along_normal * line_normal_unit
    return closest_point

def transform_plane_definition(points_and_normals, translation, rotation, logger = None):
    translated_points_and_normals = []
    for point_and_normal in points_and_normals:
        normal_and_distance = plane_6_params_to_4_params(point_and_normal)
        print(normal_and_distance)
        translated_normal_and_distance = transform_normal_and_distance(normal_and_distance, translation, rotation, logger)
        print(translated_normal_and_distance)
        translated_point_and_normal = plane_4_params_to_6_params(translated_normal_and_distance)
        print(translated_point_and_normal)
        translated_points_and_normals.append(translated_point_and_normal)
    return np.array(translated_points_and_normals)


def transform_normal_and_distance(original, translation, rotation, logger = None):
    # start_time = time.time()
    #### Build transform matrix
    # logger.info("original - {}".format(original))
    rotation_0 = np.concatenate((rotation, np.expand_dims(np.zeros(3), axis=1)), axis=1)
    translation_1 = np.array([np.concatenate((-translation, np.array([1.0])), axis=0)])
    full_transformation_matrix = np.concatenate((rotation_0, translation_1), axis=0)
    # logger.info("full_transformation_matrix - {}".format(full_transformation_matrix))

    #### Matrix multiplication
    transformed = np.transpose(np.matmul(full_transformation_matrix,original))
    # logger.info("transformed - {}".format(transformed))
    # transformed = np.array([2,0,0,6])
    print("transformed", transformed)
    normalization = np.sqrt(np.power(transformed[:3],2).sum(axis=0))
    transformed_normalized = np.concatenate((transformed[:3] / normalization, [transformed[3] * normalization]))
    # transformed_normalized = transformed
    print("transformed_normalized", transformed_normalized)
    # logger.info("transformed_normalized - {}".format(transformed_normalized))
    # logger.info("np.transpose(transformed_normalized) - {}".format(np.transpose(transformed_normalized)))
    # print("Elapsed time in geometry computes: {}".format(time.time() - start_time))
    return transformed_normalized


def transform_point(original, translation, rotation):
    original = np.array(original)
    first_group = np.concatenate((rotation, np.expand_dims( translation, axis=1)), axis=1)
    second_group = [np.array([0.,0.,0.,1.0])]
    full_transformation_matrix = np.concatenate((first_group, second_group), axis=0)
    tmp = np.empty((original.shape[0],original.shape[1]+1),dtype=original.dtype)
    tmp[:,0:3] = original
    tmp[:,3] = 1
    return full_transformation_matrix.dot(tmp.transpose())[0:3].transpose() 


# def rotation_matrix_from_euler_degrees(phi, theta, psi):
#         def Rx(phi):
#             return np.matrix([[ 1, 0           , 0           ],\
#                         [ 0, math.cos(phi),-math.sin(phi)],\
#                         [ 0, math.sin(phi), math.cos(phi)]])
        
#         def Ry(theta):
#             return np.matrix([[ math.cos(theta), 0, math.sin(theta)],\
#                         [ 0           , 1, 0           ],\
#                         [-math.sin(theta), 0, math.cos(theta)]])
        
#         def Rz(psi):
#             return np.matrix([[ math.cos(psi), -math.sin(psi), 0 ],\
#                         [ math.sin(psi), math.cos(psi) , 0 ],\
#                         [ 0           , 0            , 1 ]])

#         def degrees_to_radians(deg):
#             return deg*math.pi/180

#         R = Rz(degrees_to_radians(psi)) * Ry(degrees_to_radians(theta)) * Rx(degrees_to_radians(phi))
#         return np.array(R)


def multilist_combinations(lists):
    return list(itertools.product(*lists))

# pn = np.array([[2,0,0,1,0,0]])
# p = np.array([[1,0,0]])
# tra = np.array([1,0,0])
# rot = rotation_matrix_from_euler_degrees(0,0,0)

# print(transform_plane_definition(pn, tra, rot, None))


# print(transform_point(p, -tra, rot))

# [[0.   0.92495263 0.88753341]
# [0.92495263  0.         0.9686793 ]
# [0.88753341 0.9686793  0.        ]]


# print(((0.92495263 + 0.88753341 + 0.9686793)*2 + 3) / 3)
# 2.5955923427238172

# [[1,2,3],[4,5,6],[7,8,9,10]]
# print(multilist_combinations([[1,2,3],[4,5,6],[7,8,9,10]]))

def relative_positions(ws_1_def, ws_2_def):
    center_1, center_2 = np.array(ws_1_def["center"]), np.array(ws_2_def["center"])
    rel_pos_1 = center_2 - center_1
    rel_pos_2 = -rel_pos_1

    return rel_pos_1, rel_pos_2



def segments_distance(ws_1_def, ws_2_def):
    """ distance between two segments in the plane:
      one segment is (x11, y11) to (x12, y12)
      the other is   (x21, y21) to (x22, y22)
    """
    def segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22):
        """ whether two segments in the plane intersect:
            one segment is (x11, y11) to (x12, y12)
            the other is   (x21, y21) to (x22, y22)
        """
        dx1 = x12 - x11
        dy1 = y12 - y11
        dx2 = x22 - x21
        dy2 = y22 - y21
        delta = dx2 * dy1 - dy2 * dx1
        if delta == 0: return False  # parallel segments
        s = (dx1 * (y21 - y11) + dy1 * (x11 - x21)) / delta
        t = (dx2 * (y11 - y21) + dy2 * (x21 - x11)) / (-delta)
        return (0 <= s <= 1) and (0 <= t <= 1)

    def point_segment_distance(px, py, x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1
        if dx == dy == 0:  # the segment's just a point
            return math.hypot(px - x1, py - y1)

        # Calculate the t that minimizes the distance.
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

        # See if this represents one of the segment's
        # end points or a point in the middle.
        if t < 0:
            dx = px - x1
            dy = py - y1
        elif t > 1:
            dx = px - x2
            dy = py - y2
        else:
            near_x = x1 + t * dx
            near_y = y1 + t * dy
            dx = px - near_x
            dy = py - near_y

        return math.hypot(dx, dy)

    p11, p12, p21, p22 = ws_1_def[0], ws_1_def[1], ws_2_def[0], ws_2_def[1]
    x11, y11 = p11[0],p11[1]
    x12, y12 = p12[0],p12[1]
    x21, y21 = p21[0],p21[1]
    x22, y22 = p22[0],p22[1]

    if segments_intersect(x11, y11, x12, y12, x21, y21, x22, y22): return np.array([0])
    # try each of the 4 vertices w/the other segment
    distances = []
    distances.append(point_segment_distance(x11, y11, x21, y21, x22, y22))
    distances.append(point_segment_distance(x12, y12, x21, y21, x22, y22))
    distances.append(point_segment_distance(x21, y21, x11, y11, x12, y12))
    distances.append(point_segment_distance(x22, y22, x11, y11, x12, y12))
    return np.array([min(distances)])

def segment_intersection(segment_1, segment_2):
    a1,a2, b1,b2 = segment_1[0],segment_1[1],segment_2[0],segment_2[1],
    def perp( a ) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1