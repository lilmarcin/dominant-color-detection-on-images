# Dominant Color Detection

This project is a web application that detects and displays the 5 dominant colors present in an image. Users can provide the URL of an image, and the application will analyze the image to determine its dominant colors.

## Features

- Detects the dominant colors in an image based on a provided URL.
- Displays the original image along with the dominant color squares and their corresponding percentages.
- Allows users to easily upload and analyze images.

## Technologies Used

- Python
- Flask (Web framework)
- scikit-learn (Machine learning library)
- HTML/CSS

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/dominant-color-detection.git

2. Navigate to the project directory:

    ```bash
    cd dominant-color-detection

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt

## Usage

1. Run the Flask application:

    ```bash
    python app.py

2. Open a web browser and go to http://localhost:5000.
3. Enter the URL of an image in the provided input field.
4. Click the "Upload" button to analyze the image and display the dominant colors.

## Screenshots

![Screenshot](examples/example_1.png)

![Screenshot](examples/example_2.png)
## Acknowledgments

This project is inspired by the concept of color quantization and dominant color extraction. It utilizes the scikit-learn library for K-means clustering to identify the dominant colors in an image.

## License
This project is licensed under the MIT License.