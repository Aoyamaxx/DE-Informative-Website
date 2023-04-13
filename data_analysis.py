import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Load data
df_data = pd.read_csv('data/Annual_Surface_Temperature_Change.csv', na_values = ['-99', ''])

# Repalce column name
new_columns = {str(i): str(i) if i < 10 else str(i + 1951) for i in range(df_data.shape[1])}

# Drop some columns with useless info
df_data = df_data.drop(df_data.columns[4:10], axis = 1)

# Rename the columns of the dataframe
df_data = df_data.rename(columns=new_columns)

# Reshape the data
data_melted = pd.melt(df_data, id_vars=["ObjectId", "Country", "ISO2", "ISO3"], var_name="Year", value_name="TemperatureChange")
data_melted["Year"] = data_melted["Year"].str[1:].astype(int)  # Strip the 'F' from the year column and convert it to int

# Calculate the average temperature change for all countries
average_temperature_change = data_melted.groupby('Year')['TemperatureChange'].mean().reset_index()

# Plot the temperature change per year for all countries
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_melted, x='Year', y='TemperatureChange', hue='Country', alpha=0.2, legend=False, zorder=1)

# Plot the linear regression line for the average temperature change
sns.regplot(data=average_temperature_change, x='Year', y='TemperatureChange', color='black', label='Global Trend', ci=None, order=1, scatter_kws={'zorder': 2}, line_kws={'zorder': 2})

# Calculate the average temperature change for all countries
average_temperature_change = data_melted.groupby('Year')['TemperatureChange'].mean().reset_index()

# Plot the temperature change per year for all countries
plt.figure(figsize=(12, 6))
sns.lineplot(data=data_melted, x='Year', y='TemperatureChange', hue='Country', alpha=0.2, legend=True, zorder=1)

# Plot the linear regression line for the average temperature change
sns.regplot(data=average_temperature_change, x='Year', y='TemperatureChange', color='black', label='Global Trend', ci=None, order=1, scatter_kws={'zorder': 2}, line_kws={'zorder': 2})

# Calculate the temperature difference using the nearest available years
nearest_1961_index = average_temperature_change[average_temperature_change['Year'] >= 1961]['TemperatureChange'].notna().idxmax()
nearest_2021_index = average_temperature_change[average_temperature_change['Year'] <= 2021]['TemperatureChange'].notna().idxmin()

temp_change = average_temperature_change.loc[nearest_2021_index, 'TemperatureChange'] - \
              average_temperature_change.loc[nearest_1961_index, 'TemperatureChange']

# Display the average temperature change in the right upper corner
plt.annotate(f"1961: {average_temperature_change.loc[average_temperature_change['Year'] == 1961, 'TemperatureChange'].values[0]:.2f}°C\n"
             f"2021: {average_temperature_change.loc[average_temperature_change['Year'] == 2021, 'TemperatureChange'].values[0]:.2f}°C\n"
             f"Change: {temp_change:.2f}°C",
             xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', color='orange')

plt.title('Global Surface Temperature Change (1961-2021, in 224 Countries)')
plt.xlabel('Year')
plt.ylabel('Temperature Change')
plt.axhline(0, color='gray', linestyle='--', zorder=0)

# Remove the legend from the main plot
handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().get_legend().remove()

# Save the figure as an image
plt.savefig('surface_temperature_change_global_trend.png', dpi=300, bbox_inches='tight')

# Create a separate legend for countries
fig_legend = plt.figure(figsize=(8, 6))
ax = fig_legend.add_subplot(111)
legend = plt.legend(handles[1:], labels[1:], title='Country', loc='center', ncol=2, fontsize=10)
ax.axis('off')

# Save the separate legend as an image
fig_legend.savefig('country_legend.png', dpi=300, bbox_inches='tight')

# Generate graph for each country
if not os.path.exists('country_graphs'):
    os.makedirs('country_graphs')

countries = []
for country in data_melted['Country'].unique():

    # Filter data for the current country
    country_data = data_melted[data_melted['Country'] == country]

    nearest_1961_index_country = country_data[country_data['Year'] >= 1961]['TemperatureChange'].dropna().idxmin()
    nearest_2021_index_country = country_data[country_data['Year'] <= 2021]['TemperatureChange'].dropna().idxmax()

    temp_change_country = country_data.loc[nearest_2021_index_country, 'TemperatureChange'] - \
                      country_data.loc[nearest_1961_index_country, 'TemperatureChange']

    nearest_1961_year_country = country_data.loc[nearest_1961_index_country, 'Year']
    nearest_2021_year_country = country_data.loc[nearest_2021_index_country, 'Year']

    # Store the temperature values for the nearest available years to 1961 and 2021, and the change for the current country
    temp_1961_country = country_data.loc[nearest_1961_index_country, 'TemperatureChange']
    temp_2021_country = country_data.loc[nearest_2021_index_country, 'TemperatureChange']

    # Create a separate plot for the current country
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=country_data, x='Year', y='TemperatureChange', color='blue', zorder=1)
    sns.regplot(data=country_data, x='Year', y='TemperatureChange', color='black', ci=None, order=1, scatter_kws={'zorder': 2}, line_kws={'zorder': 2})

    # Set title and labels for the plot
    plt.title(f'Surface Temperature Change for {country} (1961-2021)')
    plt.xlabel('Year')
    plt.ylabel('Temperature Change')
    plt.axhline(0, color='gray', linestyle='--', zorder=0)

    # Display the average temperature change in the right upper corner
    plt.annotate(f"{nearest_1961_year_country}: {temp_1961_country:.2f}°C\n"
                f"{nearest_2021_year_country}: {temp_2021_country:.2f}°C\n"
                f"Change: {temp_change_country:.2f}°C",
                xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top', color='orange')

    # Replace special characters in the country name
    sanitized_country_name = country.replace(",", "").replace(".", "").replace(" ", "_")

    # Save the plot as an image in the country_graphs folder
    plt.savefig(f'country_graphs/{sanitized_country_name}_temperature_change.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    countries.append({"name": country, "image": f'country_graphs/{sanitized_country_name}_temperature_change.png'})
    
# Create a bar plot with 0 as the baseline
plt.figure(figsize=(15, 6))
bar_colors = ['#1f77b4' if temp_change < 0 else '#d62728' for temp_change in average_temperature_change['TemperatureChange']]
sns.barplot(data=average_temperature_change, x='Year', y='TemperatureChange', palette=bar_colors)

# Set title and labels for the plot
plt.title('Average Surface Temperature Change per Year (1961-2021, 224 Countries)')
plt.xlabel('Year')
plt.ylabel('Temperature Change')
plt.axhline(0, color='gray', linestyle='--', zorder=0)

# Rotate x-axis labels and display labels for every 5 years
plt.xticks(rotation=45)
xtick_labels = average_temperature_change['Year'].values
new_xtick_labels = ['' if i % 5 != 0 else str(year) for i, year in enumerate(xtick_labels)]
plt.gca().set_xticklabels(new_xtick_labels)

# Save the bar plot as an image
plt.savefig('average_temperature_change_per_year_barplot.png', dpi=300, bbox_inches='tight')

# Generate codes for country_graph.html
with open("country_graphs.html", "w") as html_file:
    html_file.write('<html>\n<head>\n<style>\n')
    html_file.write('.graph-container { display: flex; flex-wrap: wrap; }\n')
    html_file.write('.graph { flex: 1 0 50%; padding: 5px; box-sizing: border-box; }\n')
    html_file.write('.graph-row { display: flex; flex-wrap: wrap; }\n')
    html_file.write('.graph-image { width: 100%; height: auto; }\n')
    html_file.write('</style>\n</head>\n<body>\n')
    html_file.write('<div class="graph-container">\n')

    for i, country in enumerate(countries):
        if i % 2 == 0:
            html_file.write('<div class="graph-row">\n')

        html_file.write(f'<div class="graph">\n')
        html_file.write(f'<h3>{country["name"]}</h3>\n')
        html_file.write(f'<img src="{country["image"]}" alt="Temperature change for {country["name"]}" class="graph-image">\n')
        html_file.write('</div>\n')

        if i % 2 == 1 or i == len(countries) - 1:
            html_file.write('</div>\n')

    html_file.write('</div>\n')
    html_file.write('</body>\n</html>')
    
print("Task finished.")