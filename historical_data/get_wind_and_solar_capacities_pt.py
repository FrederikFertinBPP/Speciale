
import requests
import csv

# API base URL
BASE_URL = "https://servicebus.ren.pt/datahubapi/electricity/ElectricityInstalledPowerMonthly"

# Output CSV file
OUTPUT_FILE = "wind_solar_capacity_PT.csv"

# Define start and end year/month
start_year, start_month = 2025, 5
end_year, end_month = 2025, 12

# Prepare CSV header
header = ["Year", "Month", "Wind", "Solar"]

# Open CSV file for writing
with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)

    # Loop through years and months
    year, month = start_year, start_month
    while (year < end_year) or (year == end_year and month <= end_month):
        # Construct API URL
        url = f"{BASE_URL}?culture=en-US&year={year}&month={month:02d}"
        
        try:
            # Make API request
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Extract wind and solar values
            wind = None
            solar = None
            for item in data:
                if item["type"] == "WIND":
                    wind = item["monthly_Accumulation"]
                elif item["type"] == "SOLAR":
                    solar = item["monthly_Accumulation"]

            # Write row to CSV
            writer.writerow([year, month, wind, solar])
            print(f"Processed {year}-{month:02d}: Wind={wind}, Solar={solar}")

        except Exception as e:
            print(f"Error processing {year}-{month:02d}: {e}")

        # Increment month/year
        month += 1
        if month > 12:
            month = 1
            year += 1

print(f"Data saved to {OUTPUT_FILE}")
