## Introduction
This system integrates object detection with color analysis, leveraging the Ultralytics YOLOv8 model to identify objects within images and applying k-means clustering to analyze the dominant colors of those objects. The results, including object class names, dominant colors, and counts of objects with similar colors, are systematically stored in an SQLite database.

## Getting Started
### Prerequisites
- Python 3.8 or later.
- Required Python libraries: **'opencv-python'**, **'numpy'**
- Ultralytics YOLO (specifically YOLOv8) for object detection

### Installation
1. **Clone the Repository:**
First, clone this repository to your local machine:

```bash
git clone https://github.com/Abdullah-Mazhar-Arhamsoft/retail_store_shelf_product_detection.git
```


2. **Install Dependencies:**
Navigate to the project directory and install the required dependencies using the **'requirements.txt'** file:

```bash
cd retail_store_shelf_product_detection
pip install -r requirements.txt
```
This requirements.txt file includes all necessary Python packages, such as opencv-python, numpy, and ultralytics's YOLOv8.


### Running the System
To run the system, execute the script from the command line by passing the path to the image you wish to process:

```bash
python path_to_script.py /path/to/your/image.jpg
```
Ensure to replace **'path_to_script.py'** with the actual path to the Python script you're executing, and **'/path/to/your/image.jpg'** with the path to the image you want to analyze.

## System Overview
### Object Detection
Utilizes Ultralytics YOLOv8 for real-time object detection, recognizing various objects in images with high accuracy and efficiency.

### Color Analysis
Applies k-means clustering to each detected object's bounding box region to identify the dominant color(s). This method efficiently captures the primary color features within each object.

### Data Persistence
Stores analysis results in an SQLite database, **'colors_database.db'**, detailing each object's class name, the count of similar objects based on color, and the dominant color itself.

## Folder Structure
**'model/:'** Contains YOLO model files (e.g., **'yolov8m.pt'**).
**'images/:'** Directory where input images can be stored.
**'scripts/:'** Contains the Python scripts for object detection and color analysis.
**'colors_database.db:'** SQLite database file storing the analysis results.

## Usage Example
After setting up your environment according to the instructions provided:

```bash
python detect_product.py images/example.jpg
```
This command processes example.jpg, detects objects, analyzes colors, and saves the results to the database.


## License
Distributed under the MIT License. See LICENSE for more information.

Contact
Your Name - [https://github.com/Abdullah-Mazhar-Arhamsoft]

Project Link: [https://github.com/Abdullah-Mazhar-Arhamsoft/retail_store_shelf_product_detection.git]

