from flask import Flask, render_template, request, jsonify
import folium
from datetime import datetime
import requests
import json

app = Flask(__name__)

# Configuration
API_URL = "http://localhost:8000/accident-prediction"  # Update this with your actual API URL

def create_map(accident_data):
    # Create a map centered on Manhattan
    m = folium.Map(location=[40.7831, -73.9712], zoom_start=12)

    # Add accident probability markers
    for point in accident_data:
        lat = point['lat']
        lon = point['lon']
        prob = point['probability']

        # Color based on probability (red = high, yellow = medium, green = low)
        color = 'red' if prob > 0.7 else 'orange' if prob > 0.3 else 'green'

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            popup=f'Probability: {prob:.2f}',
            tooltip=f'Crash Probability: {prob:.2f}'
        ).add_to(m)

    return m._repr_html_()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get datetime from form
        date_str = request.form.get('datetime')
        if date_str:
            selected_datetime = datetime.strptime(date_str, '%Y-%m-%dT%H:%M')
        else:
            selected_datetime = datetime.now()
    else:
        # Default to current time
        selected_datetime = datetime.now()

    # Format datetime for API request
    formatted_datetime = selected_datetime.strftime('%Y-%m-%dT%H:%M:%S')

    try:
        # Make API request
        response = requests.post(
            API_URL,
            json={'date': formatted_datetime}
        )
        response.raise_for_status()
        accident_data = response.json()

        # Create map
        map_html = create_map(accident_data['predictions'])

        return render_template('index.html',
                             map_html=map_html,
                             selected_datetime=selected_datetime.strftime('%Y-%m-%dT%H:%M'))

    except Exception as e:
        return render_template('index.html',
                             error=str(e),
                             selected_datetime=selected_datetime.strftime('%Y-%m-%dT%H:%M'))

if __name__ == '__main__':
    app.run(debug=True)