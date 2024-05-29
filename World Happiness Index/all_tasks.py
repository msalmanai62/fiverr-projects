####################### task 1 ##############################
# Function to separate country data by a specific year
def separate_countries_by_year(data, year):
    year_i = []
    country_i = []
    index_i = []
    rank_i = []
    for row in data:
        cntry, yr, ind, rnk = row
        
        # Check if the row matches the specified year and index is not empty
        if yr == year and ind != '':
            country_i.append(cntry)
            year_i.append(yr)
            index_i.append(float(ind))
            rank_i.append(rnk)
    
    return country_i, year_i, index_i, rank_i

# Function to perform selection sort based on index values
def selection_sort(index_i, country_i):
    # Create a list of tuples pairing elements from lists index_i and country_i
    paired_lists = list(zip(index_i, country_i))

    # Perform a selection sort on list index_i and synchronize the sorting on list country_i in descending order
    for i in range(len(index_i)):
        max_index = i
        for j in range(i + 1, len(index_i)):
            if paired_lists[j][0] > paired_lists[max_index][0]:
                max_index = j
        
        # Swap elements in the paired list
        paired_lists[i], paired_lists[max_index] = paired_lists[max_index], paired_lists[i]

    # Extract the sorted elements from list country_i
    sorted_countries = [pair[1] for pair in paired_lists]
    return list(zip(sorted_countries, sorted(index_i, reverse=True)))

# Function to get top countries by year
def get_top_countries_by_year(data, specific_year=None, top_count=5, top_from_bottom=False, print_all=True):
    year_list = ['2013','2015','2016','2017','2018','2019','2020','2021','2022','2023']
    
    # Process for a specific year if provided and it exists in the year list
    if specific_year:
        if specific_year in year_list:
            country_i, year_i, index_i, rank_i = separate_countries_by_year(data, specific_year)
            top_countries = selection_sort(index_i, country_i)
            if top_from_bottom:
                print(f"Top {top_count} least happiest countries in year {specific_year}\n", top_countries[-top_count:], '\n\n')
            else:
                print(f"Top {top_count} most happiest countries in year {specific_year}\n", top_countries[:top_count], '\n\n')
        else:
            print("Please select year that is in this list {year_list}")
    
    # Process for all years if print_all is True and no specific year is provided
    if print_all and not specific_year:
        for year in year_list: # to print data for each year
            country_i, year_i, index_i, rank_i = separate_countries_by_year(data, year)
            top_countries = selection_sort(index_i, country_i)
            if top_from_bottom:
                print(f"Top {top_count} least happiest countries in year {year}\n", top_countries[-top_count:], '\n\n')
            else:
                print(f"Top {top_count} most happiest countries in year {year}\n", top_countries[:top_count], '\n\n')


############################ task 2 ####################
# select top_from_bottom=True if you want to print most unhappiest countries
# Function to find top countries with the most first positions in the index across 10 years
def top_3_Countries_with_most_first_positions(data, top_count=5, top_from_bottom=False):
    year_list = ['2013','2015','2016','2017','2018','2019','2020','2021','2022','2023']
    result = []

    # Iterate through each year in the year_list
    for year in year_list:
        # Retrieve country, year, index, and rank data for each year
        country_i, year_i, index_i, rank_i = separate_countries_by_year(data, year)
        
        # Perform selection sort based on the index for the current year
        top_countries = selection_sort(index_i, country_i)
        
        # Append top countries to result based on top_from_bottom condition
        if top_from_bottom:
            result.append((year, top_countries[-top_count:]))
        else:
            result.append((year, top_countries[:top_count]))

    top_in_10_years = []

    # Extract the top country in each year's top list and create a list of top countries over 10 years
    inde=0
    if top_from_bottom:
        inde = -1
    for first in result:
        top_in_10_years.append(first[1][inde][0])

    count_dict = {}
    # Count occurrences of each item in the list
    for item in top_in_10_years:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1

    # Create a list of tuples with unique items and their counts
    result_list = [(value, key) for key, value in count_dict.items()]

    print("top 3 Countries with most first positions")
    return sorted(result_list, reverse=True)[:3]

############################# task 3 ####################

# Function to find whether a country's rank has increased or decreased over a specific period
def find_country_rank(data, country, start_year, end_year):
    ranks_over_period = []
    
    # Convert start_year and end_year to integers
    start_year = int(start_year)
    end_year = int(end_year)
    
    # Iterate through the data for the specified country within the period
    for row in data:
        cntry, year, _, rank = row
        
        # Check if the row matches the specified country and falls within the specified period
        if cntry == country and start_year <= int(year) <= end_year and rank != '':
            ranks_over_period.append(int(float(rank)))
    
    if len(ranks_over_period) < 2:
        print(f"Not enough data available to analyze rank changes for {country} between {start_year} and {end_year}.")
    else:
        change = "decreasing" if ranks_over_period[-1] < ranks_over_period[0] else "increasing"
        change = change if ranks_over_period[-1] != ranks_over_period[0] else "Equal"
        print(f"The rank of {country} has been {change} over the period from {start_year} to {end_year}: {ranks_over_period[0]}->{ranks_over_period[-1]}")


################################# task 4 ###################################
        
# function to find list of countries
def list_countries(data, dsc=False): # select dsc true if you want descending order
    countries = []  
    for row in data:
        countries.append(row[0])
    return sorted(list(set(countries)), reverse=dsc)

################################# task 5 ###################################
        
# function to find countries with or above specific index value
def countries_with_index_above(data, index_threshold):
    countries_above_index = []
    for row in data:
        if row[2] and float(row[2]) >= index_threshold:  # Ensure index is available and meets the threshold
            countries_above_index.append((row[0], float(row[2])))  # Store country and index as tuple

    # Sort countries in descending order based on their index
    sorted_countries = sorted(countries_above_index, key=lambda x: x[1], reverse=True)
    return sorted_countries

################################# task 6 ###################################

# group contries contries by rank
def group_countries_by_rank_ranges(data):
    # Filter data for the last 5 years
    last_5_years_data = [row for row in data if int(row[1]) >= 2019]

    # Initialize a dictionary to store countries by rank ranges
    rank_ranges = {f"{i}-{i+9}": [] for i in range(1, 151, 10)}

    # Group countries by rank ranges for the last 5 years
    for row in last_5_years_data:
        rank = row[3]
        if rank and rank!="":  # Check if rank is available and numeric
            rank = int(float(rank))
            for start_rank in range(1, 151, 10):
                end_rank = start_rank + 9
                if start_rank <= rank <= end_rank:
                    rank_ranges[f"{start_rank}-{end_rank}"].append(row[0])
                    break  # Stop checking other ranges once added

    return rank_ranges

################################# task 7 ###################################

# countries_with_consecutive_lower_ranks over specific period
def countries_with_consecutive_lower_ranks(data, consecutive_years):
    countries = set()
    for i in range(len(data)):
        lower_count = 0
        for j in range(consecutive_years):
            # Check for missing or non-numeric rank values
            if i + j < len(data) - 1 and data[i + j][3] and data[i + j + 1][3]:
                if data[i + j][3]!="" and data[i + j + 1][3]!="":
                    if int(float(data[i + j][3])) > int(float(data[i + j + 1][3])):
                        lower_count += 1
                    else:
                        break  # Reset count if ranks are not consecutive
                else:
                    break  # Reset count if rank values are not numeric
            else:
                break  # Reset count if rank values are missing

        if lower_count == consecutive_years - 1:  # Check if consecutive lower ranks occurred
            countries.add(data[i][0])

    return list(countries)

################################# task 8 ###################################

# Function to extract details of a specific country from the dataset
def country_details(data, country_name):
    # Filter rows related to the specified country and ensure valid data for index and rank
    country_data = [row for row in data if row[0] == country_name and row[2] and row[3] 
                    and row[2] != '' and row[3] != '']

    # Check if country data is found or if data is missing/invalid
    if not country_data:
        print(f"Country '{country_name}' not found or missing data.")
        return None

    # Extract indexes and ranks from valid data rows
    indexes = [float(row[2]) for row in country_data if row[2].replace('.', '', 1) != ""]
    ranks = [int(float(row[3])) for row in country_data if row[3] != ""]

    # Check if extracted indexes or ranks are empty or invalid
    if not indexes or not ranks:
        print(f"Country '{country_name}' has invalid data.")
        return None

    # Calculate various statistics for the country
    avg_rank = sum(ranks) / len(ranks)
    rank_range = (min(ranks), max(ranks))
    index_range = (min(indexes), max(indexes))
    index_std_dev = (sum((index - avg_rank) ** 2 for index in indexes) / len(indexes)) ** 0.5
    highest_rank_year = country_data[ranks.index(max(ranks))][1]
    lowest_rank_year = country_data[ranks.index(min(ranks))][1]

    # Construct and return a dictionary containing country details
    return {
        'Country': country_name,
        'Average Rank': avg_rank,
        'Rank Range': rank_range,
        'Index Range': index_range,
        'Standard Deviation of Indexes': index_std_dev,
        'Year of Highest Rank': highest_rank_year,
        'Year of Lowest Rank': lowest_rank_year
    }

if __name__=='__main__':
    pass