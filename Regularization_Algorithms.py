import numpy as np
from scipy import optimize
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans

def detect_line(points, threshold=0.01):
    x = points[:, 0]
    y = points[:, 1]
    
    def line(x, m, b):
        return m*x + b
    
    popt, _ = optimize.curve_fit(line, x, y)
    residuals = y - line(x, *popt)
    
    return np.max(np.abs(residuals)) < threshold

def detect_circle_or_ellipse(points, threshold=0.01):
    x = points[:, 0]
    y = points[:, 1]
    
    def ellipse(x, y, xc, yc, a, b, theta):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        xct = x - xc
        yct = y - yc
        return ((xct * cos_t + yct * sin_t) / a) ** 2 + \
               ((xct * -sin_t + yct * cos_t) / b) ** 2 - 1

    def fit_func(params):
        return ellipse(x, y, *params)

    center_estimate = np.mean(points, axis=0)
    a_estimate = np.max(points[:, 0]) - np.min(points[:, 0])
    b_estimate = np.max(points[:, 1]) - np.min(points[:, 1])
    
    p0 = [center_estimate[0], center_estimate[1], a_estimate/2, b_estimate/2, 0]
    params, _ = optimize.leastsq(fit_func, p0)
    
    return np.max(np.abs(fit_func(params))) < threshold, params

def detect_rectangle_or_rounded_rectangle(points, threshold=0.01):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    if len(hull_points) != 4:
        return False, None

    edges = np.roll(hull_points, -1, axis=0) - hull_points
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angle_diffs = np.abs(np.diff(angles))
    
    is_rectangle = np.allclose(angle_diffs, np.pi/2, atol=threshold)
    
    if is_rectangle:
        return True, ("rectangle", hull_points)
    
    # Check for rounded rectangle
    corner_points = KMeans(n_clusters=4).fit(points).cluster_centers_
    distances = np.linalg.norm(points[:, None] - corner_points, axis=2)
    non_corner_points = points[np.min(distances, axis=1) > threshold]
    
    if len(non_corner_points) > 0:
        return True, ("rounded_rectangle", corner_points)
    
    return False, None

def detect_regular_polygon(points, threshold=0.01):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    edges = np.roll(hull_points, -1, axis=0) - hull_points
    edge_lengths = np.linalg.norm(edges, axis=1)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angle_diffs = np.abs(np.diff(np.concatenate([angles, [angles[0]]])))
    
    is_regular = np.allclose(edge_lengths, edge_lengths[0], atol=threshold) and \
                 np.allclose(angle_diffs, angle_diffs[0], atol=threshold)
    
    if is_regular:
        n_sides = len(hull_points)
        center = np.mean(hull_points, axis=0)
        return True, ("regular_polygon", n_sides, center, hull_points)
    
    return False, None

def detect_star_shape(points, threshold=0.01):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    center = np.mean(points, axis=0)
    radial_vectors = points - center
    distances = np.linalg.norm(radial_vectors, axis=1)
    
    peaks = distances > np.roll(distances, 1) & distances > np.roll(distances, -1)
    peak_points = points[peaks]
    
    if len(peak_points) >= 5:
        angles = np.arctan2(radial_vectors[:, 1], radial_vectors[:, 0])
        angle_diffs = np.abs(np.diff(np.sort(angles[peaks])))
        is_regular_star = np.allclose(angle_diffs, angle_diffs[0], atol=threshold)
        
        if is_regular_star:
            return True, ("star", center, peak_points)
    
    return False, None

def regularize_curve(points):
    if detect_line(points):
        return "line", np.polyfit(points[:, 0], points[:, 1], 1)
    
    is_ellipse, ellipse_params = detect_circle_or_ellipse(points)
    if is_ellipse:
        return "ellipse", ellipse_params
    
    is_rectangle, rect_params = detect_rectangle_or_rounded_rectangle(points)
    if is_rectangle:
        return rect_params
    
    is_regular_polygon, polygon_params = detect_regular_polygon(points)
    if is_regular_polygon:
        return polygon_params
    
    is_star, star_params = detect_star_shape(points)
    if is_star:
        return star_params
    
    return "irregular", points

def calculate_analytical_symmetry(curve_type, curve_params):
    if curve_type == "line":
        m, b = curve_params
        angle = np.arctan(m)
        return [angle, angle + np.pi/2]
    
    elif curve_type == "ellipse":
        xc, yc, a, b, theta = curve_params
        return [theta, theta + np.pi/2]
    
    elif curve_type == "rectangle" or curve_type == "rounded_rectangle":
        corners = curve_params[1]
        center = np.mean(corners, axis=0)
        diag1 = corners[2] - corners[0]
        diag2 = corners[3] - corners[1]
        return [np.arctan2(diag1[1], diag1[0]), np.arctan2(diag2[1], diag2[0])]
    
    elif curve_type == "regular_polygon":
        n_sides, center, vertices = curve_params[1:]
        symmetry_angles = np.linspace(0, np.pi, n_sides, endpoint=False)
        return symmetry_angles.tolist()
    
    elif curve_type == "star":
        center, peak_points = curve_params[1:]
        vectors = peak_points - center
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        symmetry_angles = np.mean(np.column_stack((angles, np.roll(angles, -1))), axis=1)
        return symmetry_angles.tolist()
    
    return []

def detect_occlusion(curve1, curve2):
    # Simple intersection check
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    
    def intersect(A, B, C, D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    for i in range(len(curve1)-1):
        for j in range(len(curve2)-1):
            if intersect(curve1[i], curve1[i+1], curve2[j], curve2[j+1]):
                return True
    return False

def complete_occluded_curve(curve, occluding_curves):
    # This is a simplified version and may need refinement
    def find_intersection(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        
        det = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if det == 0:
            return None  # Lines are parallel
        
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / det
        u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / det
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            return (x1 + t*(x2-x1), y1 + t*(y2-y1))
        return None

    intersections = []
    for occluding_curve in occluding_curves:
        for i in range(len(curve)-1):
            for j in range(len(occluding_curve)-1):
                intersection = find_intersection([curve[i], curve[i+1]], 
                                                 [occluding_curve[j], occluding_curve[j+1]])
                if intersection:
                    intersections.append(intersection)
    
    if len(intersections) < 2:
        return curve  # Not enough intersections to complete
    
    # Sort intersections based on distance from start of curve
    intersections.sort(key=lambda p: np.linalg.norm(np.array(p) - curve[0]))
    
    # Create new curve by connecting intersections
    new_curve = np.array([curve[0]] + intersections + [curve[-1]])
    
    return new_curve

def process_occlusions(curves):
    completed_curves = []
    for i, curve in enumerate(curves):
        occluding_curves = [c for j, c in enumerate(curves) if i != j and detect_occlusion(curve, c)]
        if occluding_curves:
            completed_curve = complete_occluded_curve(curve, occluding_curves)
            completed_curves.append(completed_curve)
        else:
            completed_curves.append(curve)
    return completed_curves