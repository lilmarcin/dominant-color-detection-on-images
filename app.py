from flask import Flask, render_template, send_from_directory, request
from flask_uploads import UploadSet, IMAGES, configure_uploads

import os
import requests
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdfgh'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

def is_color_light(rgb):
    r, g, b = rgb
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return brightness > 127


def save_image_from_url(image_url):
    filename = os.path.basename(image_url)
    filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)

    response = requests.get(image_url)
    with open(filepath, 'wb') as out_file:
        out_file.write(response.content)

    return filepath

def get_dominant_colors(image_path, num_colors):
    image = Image.open(image_path)
    image = image.convert("RGB")

    image_array = np.array(image)

    flattened_array = image_array.reshape(-1, 3)

    # Wykorzystanie algorytmu k-means do grupowania pikseli
    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(flattened_array)

    # Uzyskanie etykiet klastrów
    labels = kmeans.labels_

    # Zliczenie wystąpień etykiet
    label_counts = Counter(labels)

    # Sortowanie etykiet według liczby wystąpień
    sorted_labels = sorted(label_counts, key=lambda x: label_counts[x], reverse=True)

    # Uzyskanie centrum klastrów
    cluster_centers = kmeans.cluster_centers_

    # Obliczenie udziału procentowego dla każdego koloru
    total_pixels = flattened_array.shape[0]
    colors_percentages = [(label_counts[i] / total_pixels) * 100 for i in sorted_labels]

    # Uzyskanie dominujących kolorów
    dominant_colors = [cluster_centers[i] for i in sorted_labels]

    # Konwersja wartości kolorów do typu całkowitoliczbowego
    dominant_colors = [tuple(map(int, color)) for color in dominant_colors]

    # Zwrócenie listy dominujących kolorów wraz z ich udziałem procentowym
    result = [(color, int(percentage)) for color, percentage in zip(dominant_colors, colors_percentages)]
    return result[:5]


@app.route('/', methods=['GET', 'POST'])
def display_image():
    if request.method == 'POST':
        image_url = request.form.get('image_url')
        image_filepath = save_image_from_url(image_url)

        # Pobieranie 5 dominujących kolorów
        num_colors = 5
        dominant_colors = get_dominant_colors(image_filepath, num_colors)

        return render_template('index.html', image_url=image_url, dominant_colors=dominant_colors, is_color_light=is_color_light)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True,host='127.0.0.1', port=5000)
