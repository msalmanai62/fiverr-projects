import os
import sys
from all_tasks import get_top_countries_by_year, top_3_Countries_with_most_first_positions, find_country_rank, list_countries, countries_with_index_above, group_countries_by_rank_ranges, countries_with_consecutive_lower_ranks, country_details

def read_data(file_path):
    if not os.path.isfile(file_path):
        print("File not found.")
        sys.exit()
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data.append(row)
    return data

def select_task(file_path):
    data = read_data(file_path)
    instructions = """
        Please select from the menu to perform a specific operation:
        1. Top 10 happiest countries or least happiest.
        2. Top 3 countries with most first positions from top and bottom.
        3. Specific country with increasing or decreasing rank over a specific period.
        4. Find a list of countries.
        5. Countries with or above a specific index value.
        6. Group countries by rank.
        7. Countries with consecutive lower ranks over a specific period.
        8. Specific country details.
    """
    print(instructions)
    try:
        choice = int(input("Please select a number from 1 to 8 to perform an operation: "))
    except ValueError:
        print("Invalid input. Please enter a number.")
        return

    header = data[0]
    data_ = data[1:]

    if choice == 1:
        specific_year = input("Enter year: ")
        top_from_bottom = input("Select top_from_bottom value (True/False): ").lower() == "true"
        top_count = int(input("Enter top count: "))
        get_top_countries_by_year(data_, specific_year=specific_year, top_from_bottom=top_from_bottom, top_count=top_count, print_all=False)

    elif choice == 2:
        top_from_bottom = input("Select top_from_bottom value (True/False): ").lower() == "true"
        print(top_3_Countries_with_most_first_positions(data_, top_from_bottom=top_from_bottom, top_count=3))

    elif choice == 3:
        country = input("Enter country name (from the list): ")
        try:
            start_year = int(input("Enter Start Year: "))
            end_year = int(input("Enter End Year: "))
            find_country_rank(data, country, start_year, end_year)
        except ValueError:
            print("Invalid year format. Please enter a number.")

    elif choice == 4:
        dsc = input("Descending order? (True/False): ").lower() == "true"
        print(list_countries(data_, dsc=dsc))

    elif choice == 5:
        try:
            index_threshold = float(input("Enter threshold value (float): "))
            result_countries = countries_with_index_above(data_, index_threshold)
            for country, index in result_countries:
                print(f"{country}: {index}")
        except ValueError:
            print("Invalid threshold value. Please enter a float number.")

    elif choice == 6:
        print(group_countries_by_rank_ranges(data_))

    elif choice == 7:
        try:
            consecutive_years = int(input("Enter the number of consecutive years: "))
            countries_with_consecutive_lower = countries_with_consecutive_lower_ranks(data_, consecutive_years)
            print(f"Countries with at least {consecutive_years} consecutive years of lower ranks: {countries_with_consecutive_lower}")
        except ValueError:
            print("Invalid input. Please enter a number.")

    elif choice == 8:
        country_name = input("Enter Country name: ")
        print(country_details(data_, country_name))

    else:
        print("Please select a valid choice (1 to 8).")

if __name__ == "__main__":
    file_path = 'world_happiness_index_2013_2023.csv'
    select_task(file_path)
