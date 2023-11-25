from flask import Flask, render_template, url_for
import os
import os
from jinja2 import Template  # Import the Jinja template engine


app = Flask(__name__)


def display_images():
    image_folder = '.\\static'  # Replace with the path to your image folder

    image_files = []
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_files.append(filename)

    #Reverse the list so that the latest image is displayed first
    image_files.reverse()

    return render_template('index.html', image_files=image_files)

@app.route('/')
def index():
    return display_images()


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
