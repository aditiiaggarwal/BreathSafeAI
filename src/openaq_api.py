
import os
import math
import csv
import requests
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv


# ==========================================
# 1. LOAD API KEY
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("OPENAQ_API_KEY")

OPENAQ_URL = "https://api.openaq.org/v3"

GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)


# ==========================================
# 2. CALCULATE DISTANCE
# ==========================================

def calculate_distance(lat1, lon1, lat2, lon2):

    radius = 6371

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return radius * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


# ==========================================
# 3. SEARCH FOR ANY CITY
# ==========================================

def search_city(city_name):

    params = {
        "name": city_name,
        "count": 10,
        "language": "en",
        "format": "json"
    }

    try:

        response = requests.get(
            GEOCODING_URL,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            print("City not found. Try another city.")
            return None

        print("\nAVAILABLE LOCATIONS")
        print("=" * 50)

        for index, place in enumerate(results, start=1):

            print(
                f"{index}. {place['name']}, "
                f"{place.get('admin1', '')}, "
                f"{place.get('country', '')}"
            )

        while True:

            try:

                choice = int(
                    input("\nSelect your city number: ")
                )

                if 1 <= choice <= len(results):
                    return results[choice - 1]

                print("Invalid selection. Try again.")

            except ValueError:

                print("Please enter a valid number.")

    except requests.exceptions.RequestException as error:

        print("Geocoding error:", error)
        return None


# ==========================================
# 4. FIND MONITORING STATIONS
# ==========================================

def get_locations(latitude, longitude, country_code):

    if not API_KEY:
        print("ERROR: OpenAQ API key not found!")
        return []

    headers = {
        "X-API-Key": API_KEY
    }

    params = {
        "bbox": (
            f"{longitude - 1.5},"
            f"{latitude - 1.5},"
            f"{longitude + 1.5},"
            f"{latitude + 1.5}"
        ),
        "iso": country_code,
        "limit": 1000,
        "page": 1
    }

    try:

        response = requests.get(
            f"{OPENAQ_URL}/locations",
            headers=headers,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        locations = response.json().get("results", [])

        for location in locations:

            coordinates = location.get("coordinates", {})

            station_lat = coordinates.get("latitude")
            station_lon = coordinates.get("longitude")

            if station_lat is None or station_lon is None:

                location["calculated_distance"] = float("inf")
                continue

            location["calculated_distance"] = calculate_distance(
                latitude,
                longitude,
                station_lat,
                station_lon
            )

        locations.sort(
            key=lambda x: x["calculated_distance"]
        )

        print("\nMONITORING STATIONS NEAR YOUR CITY")
        print("=" * 50)

        if not locations:

            print("No monitoring stations found in this area.")
            return []

        for index, location in enumerate(locations[:20], start=1):

            print(f"\n{index}. {location.get('name')}")

            print("Location ID:", location.get("id"))

            print(
                "Distance:",
                round(location["calculated_distance"], 2),
                "km"
            )

            print("Available pollutants:")

            for sensor in location.get("sensors", []):

                parameter = sensor.get("parameter", {})

                print(
                    "-",
                    parameter.get("name"),
                    "| Unit:",
                    parameter.get("units")
                )

        return locations

    except requests.exceptions.RequestException as error:

        print("OpenAQ location error:", error)
        return []


# ==========================================
# 5. FETCH LATEST MEASUREMENTS AND SAVE CSV
# ==========================================

def get_latest_measurements(location, city_name):

    headers = {
        "X-API-Key": API_KEY
    }

    location_id = location.get("id")

    try:

        sensor_details = {}

        for sensor in location.get("sensors", []):

            parameter = sensor.get("parameter", {})

            sensor_details[sensor.get("id")] = {
                "name": parameter.get("name", "Unknown"),
                "unit": parameter.get("units", "")
            }

        response = requests.get(
            f"{OPENAQ_URL}/locations/{location_id}/latest",
            headers=headers,
            params={
                "limit": 100,
                "page": 1
            },
            timeout=20
        )

        response.raise_for_status()

        measurements = response.json().get("results", [])

        print("\nLATEST AIR QUALITY MEASUREMENTS")
        print("=" * 50)

        if not measurements:

            print("No latest measurements available.")
            return []

        # Create data folder and CSV file

        data_folder = BASE_DIR / "data"

        data_folder.mkdir(parents=True, exist_ok=True)

        csv_file = data_folder / "live_measurements.csv"

        file_exists = csv_file.exists()

        with open(
            csv_file,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            # Write headings only once

            if not file_exists or csv_file.stat().st_size == 0:

                writer.writerow([
                    "city",
                    "station",
                    "location_id",
                    "distance_km",
                    "pollutant",
                    "value",
                    "unit",
                    "timestamp",
                    "collected_at"
                ])

            for reading in measurements:

                sensor_id = reading.get("sensorsId")

                details = sensor_details.get(sensor_id, {})

                pollutant = details.get("name", "Unknown")

                unit = details.get("unit", "")

                value = reading.get("value")

                datetime_info = reading.get("datetime", {})

                timestamp = datetime_info.get("local", "")

                print("\nPollutant:", pollutant)

                print("Value:", value, unit)

                print("Time:", timestamp)

                # Save measurement to CSV

                writer.writerow([
                    city_name,
                    location.get("name"),
                    location_id,
                    round(location.get("calculated_distance", 0), 2),
                    pollutant,
                    value,
                    unit,
                    timestamp,
                    datetime.now().astimezone().isoformat()
                ])

        print("\nData successfully saved!")

        print("File:", csv_file)

        return measurements

    except requests.exceptions.RequestException as error:

        print("Measurement error:", error)
        return []


# ==========================================
# 6. MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 50)
    print("              BREATHSAFE AI")
    print("       AIR QUALITY MONITORING SYSTEM")
    print("=" * 50)

    city = input(
        "\nEnter city name (e.g. Laxmangarh, Rajasthan): "
    ).strip()

    if not city:

        print("Please enter a city name.")
        raise SystemExit

    selected_city = search_city(city)

    if selected_city is None:
        raise SystemExit

    city_name = selected_city.get("name")
    country = selected_city.get("country")
    country_code = selected_city.get("country_code")

    latitude = selected_city.get("latitude")
    longitude = selected_city.get("longitude")

    print("\nSELECTED CITY")
    print("=" * 50)

    print("City:", city_name)
    print("Country:", country)
    print("Latitude:", latitude)
    print("Longitude:", longitude)

    locations = get_locations(
        latitude,
        longitude,
        country_code
    )

    if not locations:

        print("\nNo monitoring stations found.")
        raise SystemExit

    usable_locations = [
        location for location in locations
        if location.get("sensors")
    ]

    if not usable_locations:

        print("\nNo stations with available sensor information.")
        raise SystemExit

    selected_station = usable_locations[0]

    print("\nSELECTED MONITORING STATION")
    print("=" * 50)

    print("Station:", selected_station.get("name"))

    print(
        "Distance:",
        round(selected_station["calculated_distance"], 2),
        "km"
    )

    # Fetch readings and save them to CSV

    get_latest_measurements(selected_station, city_name)

    print("\n" + "=" * 50)
    print("          BREATHSAFE AI FINISHED")
    print("=" * 50)
