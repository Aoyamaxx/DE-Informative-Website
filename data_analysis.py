import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

df_data = pd.read_csv('data/Annual_Surface_Temperature_Change.csv', na_values = ['-99', ''])

# repalce column name
new_columns = {str(i): str(i) if i < 10 else str(i + 1951) for i in range(df_data.shape[1])}

# drop some columns with useless info
df_data = df_data.drop(df_data.columns[4:10], axis = 1)

# rename the columns of the dataframe
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

# Calculate the temperature difference
temp_change = average_temperature_change.loc[average_temperature_change['Year'] == 2021, 'TemperatureChange'].values[0] - \
              average_temperature_change.loc[average_temperature_change['Year'] == 1961, 'TemperatureChange'].values[0]

# Display the average temperature change in the right upper corner
plt.annotate(f"1961: {average_temperature_change.loc[average_temperature_change['Year'] == 1961, 'TemperatureChange'].values[0]:.2f}°C\n"
             f"2021: {average_temperature_change.loc[average_temperature_change['Year'] == 2021, 'TemperatureChange'].values[0]:.2f}°C\n"
             f"Change: {temp_change:.2f}°C",
             xy=(0.95, 0.95), xycoords='axes fraction', fontsize=12, ha='right', va='top')

plt.title('Surface Temperature Change (1961-2021)')
plt.xlabel('Year')
plt.ylabel('Temperature Change')

# Get the legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()

# Remove the legend from the main plot
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

# Loop through all unique countries in the dataset
for country in data_melted['Country'].unique():

    # Filter data for the current country
    country_data = data_melted[data_melted['Country'] == country]

    # Create a separate plot for the current country
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=country_data, x='Year', y='TemperatureChange', color='blue', zorder=1)
    sns.regplot(data=country_data, x='Year', y='TemperatureChange', color='black', ci=None, order=1, scatter_kws={'zorder': 2}, line_kws={'zorder': 2})

    # Set title and labels for the plot
    plt.title(f'Surface Temperature Change for {country} (1961-2021)')
    plt.xlabel('Year')
    plt.ylabel('Temperature Change')

    # Replace special characters in the country name
    sanitized_country_name = country.replace(",", "").replace(".", "").replace(" ", "_")

    # Save the plot as an image in the country_graphs folder
    plt.savefig(f'country_graphs/{sanitized_country_name}_temperature_change.png', dpi=300, bbox_inches='tight')

    plt.close()