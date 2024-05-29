######### Libraries and thier installation guide
# we are using following libraries in for this assignment
# pandas  --> pip install pandas
# numpy ---> pip install numpy
# matplotlib --> pip install matplotlib
# sklearn --> pip install scikit-learn

######################## Task 1 #####################
# reading data
import pandas as pd # importing pandas library
data = pd.read_csv('stock_data.csv', sep=',') # reading our CSV data file
# drop the unwanted columns and just keep date, close, Name columns
data = data.drop(['open', 'high', 'low', 'volume'], axis=1)


######################## Task 2 #####################
# finding number of unique names
# Get unique names from a column and sort them alphabetically
unique_names = sorted(data['Name'].unique())  # Collect unique names and sort them alphabetically
num_names = len(unique_names)  # Count the number of unique names
first_5_names = unique_names[:5]  # Select the first five names from the sorted list
last_5_names = unique_names[-5:]  # Select the last five names from the sorted list

# Output the results
print("Number of unique names:", num_names)
print("First 5 names:", first_5_names)
print("Last 5 names:", last_5_names)


######################## Task 3 #####################
# calculating num_remaining_names and removed_names

# Convert 'Date' column to datetime format
data['date'] = pd.to_datetime(data['date'])  # Convert the 'date' column to a datetime format
# Filter names based on date criteria
filtered_data = data.groupby('Name').filter(
    lambda x: (x['date'].min() <= pd.Timestamp('2014-07-01')) and (x['date'].max() >= pd.Timestamp('2017-06-30'))
)  # Select names based on date criteria: between July 1, 2014, and June 30, 2017
# Find removed names and count remaining unique names
removed_names = sorted(set(data['Name'].unique()) - set(filtered_data['Name'].unique()))  # Names removed due to date criteria
num_remaining_names = len(filtered_data['Name'].unique())  # Count of remaining unique names
# Output the results
print("Names removed:", removed_names)
print("Number of remaining names:", num_remaining_names)


######################## Task 4 #####################
# Filtering dates based on specific criteria

filtered_dates = filtered_data[
    (filtered_data['date'] >= pd.Timestamp('2014-07-01')) & (filtered_data['date'] <= pd.Timestamp('2017-06-30'))
]  # Select dates between July 1, 2014, and June 30, 2017
# Find unique dates and count them
unique_dates = sorted(filtered_dates['date'].unique())  # Collect unique dates and sort them
num_dates = len(unique_dates)  # Count the number of unique dates
# Select the first five and last five dates
first_5_dates = unique_dates[:5]  # Get the first five dates
last_5_dates = unique_dates[-5:]  # Get the last five dates
# Output the results
print("Number of dates:", num_dates)
print("First 5 dates:", first_5_dates) 
print("Last 5 dates:", last_5_dates)


######################## Task 5 #####################
# pivot table for close values

# Build a new DataFrame with close values for each name and date
pivot_data = filtered_data.pivot(index='date', columns='Name', values='close')
print(pivot_data)


######################## Task 6 #####################
# returns calculatation

returns_data = {}  # Initialize an empty dictionary to store returns for each stock
# Loop through unique 'Name's in filtered_data
for name in filtered_data['Name'].unique():
    temp = filtered_data[filtered_data['Name'] == name].sort_values('date')  # Select data for each 'Name' and sort by 'date'
    temp['Return'] = (temp['close'] - temp['close'].shift(1)) / temp['close'].shift(1)  # Calculate daily returns
    returns_data[name] = temp['Return'].values[1:]  # Store calculated returns in the dictionary
returns_data_2 = {}  # Initialize another dictionary for filtered returns
# Loop through items in returns_data
for item, value in returns_data.items():
    if len(value) == 1258:  # Check if the number of returns is 1258 (assuming daily data for about 5 years)
        returns_data_2[item] = value  # Store returns with correct length in another dictionary
returns_df = pd.DataFrame(returns_data_2)  # Create a DataFrame using the filtered returns
print(returns_df)


######################## Task 7 #####################
# calculating the principal components of the return

from sklearn.decomposition import PCA
# Calculate principal components
pca = PCA()  # Initialize PCA
pca.fit(returns_df.dropna())  # Fit PCA model after dropping NaN values in returns_df
# Print top five principal components by eigenvalue
top_five_PCs = pca.components_[:5]  # Extract the top five principal components
print("Top five principal components:")
for i, pc in enumerate(top_five_PCs, 1):  # Iterate through top five principal components
    print(f"PC {i}: {pc}")  # Print each principal component


######################## Task 8 #####################
# Calculating and plotting the cumulative variance ratios

import matplotlib.pyplot as plt
# Extract explained variance ratios from PCA
explained_variance_ratios = pca.explained_variance_ratio_  # Extract explained variance ratios
# Calculate percentage of variance explained by the first principal component
variance_explained = explained_variance_ratios[0] * 100  # Calculate variance explained by the first PC
print(f"Percentage of variance explained by the first PC: {variance_explained:.2f}%")
# Plot explained variance ratios for the top 20 principal components
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(range(1, 21), explained_variance_ratios[:20], marker='o')  # Plot explained variance ratios for the top 20 PCs
plt.xlabel('Principal Component')  # Set x-axis label
plt.ylabel('Explained Variance Ratio')  # Set y-axis label
plt.title('Explained Variance Ratios for Top 20 PCs')  # Set plot title
plt.show()  # Display the plot


######################## Task 9 #####################
# explained variance ratios calculation and plotting

import numpy as np
# Calculate cumulative variance ratios
cumulative_variance_ratios = np.cumsum(explained_variance_ratios)  # Calculate cumulative sum of explained variance ratios
# Plot cumulative variance ratios
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(range(1, len(cumulative_variance_ratios) + 1), cumulative_variance_ratios, marker='o')  # Plot cumulative variance ratios
plt.xlabel('Principal Component')  # Set x-axis label
plt.ylabel('Cumulative Variance Ratio')  # Set y-axis label
plt.title('Cumulative Variance Ratios')  # Set plot title
plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')  # Add a horizontal line at 95% explained variance
plt.legend()  # Show the legend
plt.show()  # Display the plot


######################## Task 10 #####################
# # Plotting cumulative variance ratios for normalized data

from sklearn.preprocessing import StandardScaler  # Import StandardScaler for normalization
# Normalizing the returns DataFrame
scaler = StandardScaler()  # Initialize StandardScaler
normalized_returns = scaler.fit_transform(returns_df.dropna())  # Normalize the returns data after dropping NaN values
# Applying PCA to normalized data
pca_normalized = PCA()  # Initialize PCA for normalized data
pca_normalized.fit(normalized_returns)  # Fit PCA to the normalized returns data
# Extract explained variance ratios for normalized data
explained_variance_ratios_normalized = pca_normalized.explained_variance_ratio_  # Extract explained variance ratios
# Calculate cumulative variance ratios for normalized data
cumulative_variance_ratios_normalized = np.cumsum(explained_variance_ratios_normalized)  # Calculate cumulative variance ratios
# Plot cumulative variance ratios for normalized data
plt.figure(figsize=(8, 6))  # Set the figure size
plt.plot(range(1, len(cumulative_variance_ratios_normalized) + 1), cumulative_variance_ratios_normalized, marker='o')  # Plot cumulative variance ratios
plt.xlabel('Principal Component')  # Set x-axis label
plt.ylabel('Cumulative Variance Ratio')  # Set y-axis label
plt.title('Cumulative Variance Ratios (Normalized Data)')  # Set plot title
plt.axhline(y=0.95, color='r', linestyle='--', label='95% explained variance')  # Add a horizontal line at 95% explained variance
plt.legend()  # Show the legend
plt.show()  # Display the plot
