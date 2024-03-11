import cv2
import numpy as np

def extract_bbox_coordinates(yolo_bbox, image_width, image_height):
    """
    The function `extract_bbox_coordinates` converts YOLO bounding box coordinates to image coordinates.
    
    :param yolo_bbox: The `yolo_bbox` parameter represents the bounding box coordinates in YOLO format.
    It typically consists of four values: x_center, y_center, width, and height
    :param image_width: The `image_width` parameter represents the width of the image in pixels. It is
    used in the `extract_bbox_coordinates` function to calculate the bounding box coordinates based on
    the YOLO format bounding box and the image dimensions
    :param image_height: The `image_height` parameter represents the height of the image in pixels. It
    is used in the `extract_bbox_coordinates` function to calculate the bounding box coordinates based
    on the YOLO format bounding box and the dimensions of the image
    :return: The function `extract_bbox_coordinates` returns the coordinates of the bounding box in the
    format (x1, y1, x2, y2) where:
    - x1: The x-coordinate of the top-left corner of the bounding box
    - y1: The y-coordinate of the top-left corner of the bounding box
    - x2: The x-coordinate of the bottom-right corner of the
    """

    try:
        x_center, y_center, width, height = yolo_bbox
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)
        return x1, y1, x2, y2
    
    except Exception as e:
        print(f"An error occurred in extract_bbox_coordinates: {e}")
        return None

def find_dominant_colors(image, k=1):
    """
    The function `find_dominant_colors` uses k-means clustering to identify the dominant colors in an
    image.
    
    :param image: The `image` parameter in the `find_dominant_colors` function is expected to be a NumPy
    array representing an image. The function reshapes the image array to have a shape of (-1, 3), where
    -1 indicates that the size of that dimension is inferred based on the
    :param k: The `k` parameter in the `find_dominant_colors` function represents the number of dominant
    colors you want to extract from the image using the K-means clustering algorithm. By default, it is
    set to 1, meaning the function will find the most dominant color in the image. If, defaults to 1
    (optional)
    :return: The function `find_dominant_colors` returns the dominant color(s) in the image based on the
    specified value of k (number of dominant colors to find). In this case, with k=1, it returns the
    single dominant color as a numpy array representing the BGR values of that color.
    """

    try:
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        return centers[0]
    
    except Exception as e:
        print(f"An error occurred in find_dominant_colors: {e}")
        return None

def color_distance(color1, color2):
    """
    The function calculates the Euclidean distance between two colors represented as arrays.
    
    :param color1: It seems like you were about to provide some information about the `color1`
    parameter. Could you please provide more details or specify the values for `color1` so that I can
    assist you further with the `color_distance` function?
    :param color2: It seems like you have provided the function definition for calculating the distance
    between two colors using the Euclidean distance formula. However, you have not provided the
    definition or values for `color2`. Could you please provide the values for `color2` so that I can
    help you calculate the color distance between
    :return: The function `color_distance` calculates the Euclidean distance between two colors
    represented as arrays. It returns the Euclidean distance between `color1` and `color2`.
    """
    try:
        return np.sqrt(np.sum((color1 - color2) ** 2))
    
    except Exception as e:
        print(f"An error occurred in color_distance: {e}")
        return None

def find_similar_objects(dominant_colors):
    """
    The function `find_similar_objects` takes a list of dominant colors, groups similar colors based on
    a similarity threshold, and returns a dictionary with counts of similar colors.
    
    :param dominant_colors: It seems like you were about to provide the `dominant_colors` parameter for
    the `find_similar_objects` function. Please go ahead and provide the list of dominant colors so that
    I can assist you further with the code
    :return: The function `find_similar_objects` returns a dictionary `color_groups` where the keys are
    tuples representing dominant colors and the values are dictionaries containing the index of the
    color in the input list `dominant_colors` and the count of similar colors found based on a
    similarity threshold of 10.
    """

    try:
        color_groups = {}
        for i, color in enumerate(dominant_colors):
            found_similar = False
            for known_color in color_groups.keys():
                if color_distance(color, known_color) < 10:
                    color_groups[known_color]['count'] += 1
                    found_similar = True
                    break
            if not found_similar:
                color_groups[tuple(color)] = {'index': i, 'count': 1}
        return color_groups
    
    except Exception as e:
        print(f"An error occurred in find_similar_objects: {e}")
        return None

def colors_extracted(image, objects_data):
    """
    The function `colors_extracted` processes object data from an image to extract dominant colors and
    group objects by color before compiling the results.
    
    :param image: The `image` parameter is a NumPy array representing an image, with shape (height,
    width, channels)
    :param objects_data: The `objects_data` parameter is a list containing data for each object detected
    in the image. Each element in the list represents an object and consists of the following
    information:
    :return: The function `colors_extracted` returns a list of dictionaries, where each dictionary
    contains information about the objects grouped by dominant color. Each dictionary includes the class
    ID of the object, the count of objects with that color, and the dominant color itself.
    """

    try:
        image_height, image_width, _ = image.shape

        # Process object data and extract dominant colors
        dominant_colors = []
        for obj_data in objects_data:
            _, bbox = obj_data[0], obj_data[1:]
            x1, y1, x2, y2 = extract_bbox_coordinates(bbox, image_width, image_height)
            cropped_region = image[y1:y2, x1:x2]
            dominant_color = find_dominant_colors(cropped_region, k=1)
            dominant_colors.append(dominant_color)

        # Group objects by color
        color_groups = find_similar_objects(dominant_colors)

        # Compile results
        results = []
        for color, info in color_groups.items():
            obj_index = info['index']
            count = info['count']
            class_id = objects_data[obj_index][0]
            results.append({'class_id': class_id, 'count': count, 'color': color})

        return results
    except Exception as e:
        print(f"An error occurred in colors_extracted: {e}")
        return None

