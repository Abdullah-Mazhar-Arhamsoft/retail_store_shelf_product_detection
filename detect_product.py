from ultralytics import YOLO
import cv2
from extract_color import colors_extracted
import sqlite3
import argparse

def initialize_model():
    """
    The function initializes a YOLO model using the specified model path.
    :return: An instance of the YOLO class initialized with the model path "model/yolov8m.pt" is being
    returned.
    """
    model_path = "model/yolov8m.pt"
    return YOLO(model_path)

def predict_objects(model, image):
    """
    The function `predict_objects` takes a model and an image as input, predicts objects in the image
    using the model, and returns the bounding boxes with class IDs and class names.
    
    :param model: The `model` parameter in the `predict_objects` function is expected to be an object
    that has a `predict` method. This method takes an image as input and returns the results of the
    prediction, which include information about the detected objects such as bounding boxes, class IDs,
    and class names
    :param image: The `predict_objects` function takes a model and an image as input parameters. The
    model is used to make predictions on the provided image to detect objects within it. The function
    processes the results of the prediction and extracts information about the detected objects such as
    bounding box coordinates, class IDs, and class names
    :return: The function `predict_objects` returns a list of detections containing the class ID,
    x_center, y_center, width, and height for each detected object in the image, as well as the class
    names associated with each class ID.
    """

    try:
        results = model.predict(source=image)
        boxes_xywhn = results[0].boxes.xywhn
        class_ids = results[0].boxes.cls
        class_names = results[0].names

        # Pack bounding boxes with class IDs
        detections = []
        for box, class_id in zip(boxes_xywhn.numpy(), class_ids.numpy().astype(int)):
            x_center, y_center, width, height = box[:4]
            detections.append([class_id, x_center, y_center, width, height])

        all_detections = list(detections)
        
        return all_detections, class_names
    
    except Exception as e:
        print(f"Error predicting objects: {e}")
        return None, None

def replace_class_ids_with_names(colors_result, class_names):
    """
    The function `replace_class_ids_with_names` replaces class IDs with corresponding class names in a
    list of dictionaries.
    
    :param colors_result: colors_result is a list of dictionaries where each dictionary represents a
    color result with keys like 'class_id' and values associated with it
    :param class_names: The `class_names` parameter is a dictionary that maps class IDs to their
    corresponding class names. It is used in the `replace_class_ids_with_names` function to replace
    class IDs in the `colors_result` list with their respective class names
    :return: The function `replace_class_ids_with_names` returns the `colors_result` list after
    replacing the class IDs with corresponding class names from the `class_names` dictionary. Each item
    in the `colors_result` list will have a new key 'class_name' with the corresponding class name, or
    'Unknown' if the class ID is not found in the `class_names` dictionary. The original 'class
    """

    for item in colors_result:
        class_id = item['class_id']
        if class_id in class_names:
            item['class_name'] = class_names[class_id]
        else:
            item['class_name'] = 'Unknown'
        del item['class_id']

    return colors_result

def save_to_database(colors_result_with_names, database_path):
    """
    The function `save_to_database` saves color results with names to a SQLite database.
    
    :param colors_result_with_names: The `colors_result_with_names` parameter seems to be a list of
    dictionaries where each dictionary represents a color result with the following keys: 'class_name',
    'count', and 'color'
    :param database_path: The `database_path` parameter is a string that represents the path to the
    SQLite database file where you want to save the data. This function `save_to_database` takes two
    parameters: `colors_result_with_names` which is a list of dictionaries containing information about
    colors, and `database_path` which
    """

    try:
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute('''CREATE TABLE IF NOT EXISTS colors_results (
                            class_name TEXT,
                            quantity INTEGER,
                            color TEXT
                        )''')

        # Insert data into the table
        for item in colors_result_with_names:
            class_name = item['class_name']
            quantity = item['count']
            color = item['color']
            cursor.execute("INSERT INTO colors_results (class_name, quantity, color) VALUES (?, ?, ?)", (class_name, quantity, str(color)))

        # Commit changes and close connection
        conn.commit()
        conn.close()

    except Exception as e:
        print(f"Error saving to database: {e}")


def process_image(image_path):
    """
    The function processes an image by predicting objects, extracting colors, replacing class IDs with
    names, and saving the results to a database.
    
    :param image_path: The `image_path` parameter is a string that represents the file path to the image
    that you want to process. This image will be read using OpenCV (`cv2`) and then used for object
    detection and color extraction in the `process_image` function
    """

    try:
        model = initialize_model()
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to read the image.")
            return

        boxes, class_names = predict_objects(model, image)

        if boxes is None or class_names is None:
            print("Error: Object detection failed.")
            return

        colors_result = colors_extracted(image, boxes)
        colors_result_with_names = replace_class_ids_with_names(colors_result, class_names)

        database_path = "colors_database.db"
        save_to_database(colors_result_with_names, database_path)
        
    except Exception as e:
        print(f"Error processing image: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image to detect objects and save color results to a database.")
    parser.add_argument("image", help="Path to the input image file")
    args = parser.parse_args()

    process_image(args.image)
