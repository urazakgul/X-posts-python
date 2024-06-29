import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

def fetch_euro_squads(year):
    url = f"https://en.wikipedia.org/wiki/UEFA_Euro_{year}_squads"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tables = pd.read_html(url)

    headers = soup.find_all('h3')
    all_tables = []

    max_tables = 24 if year >= 2016 else 16

    for i, (header, table) in enumerate(zip(headers, tables)):
        if i >= max_tables:
            break

        country_id = header.find('span', id=True)['id']
        table.insert(0, 'Country', country_id)
        all_tables.append(table)

    merged_df = pd.concat(all_tables, ignore_index=True)
    merged_df['Year'] = year

    return merged_df

years = list(range(2024, 1995, -4))

all_data = []
for year in years:
    df = fetch_euro_squads(year)
    all_data.append(df)

final_df = pd.concat(all_data, ignore_index=True)

def extract_age(dob_age):
    if isinstance(dob_age, str):
        return int(dob_age.split()[-1].strip('()aged'))
    return None

final_df['Age'] = final_df['Date of birth (age)'].apply(extract_age)

age_stats = final_df.groupby(['Year', 'Country'])['Age'].agg(['mean', 'std']).reset_index()

age_stats_2024 = age_stats[age_stats['Year'] == 2024]

highlight_countries = [
    "Germany", "Spain", "Portugal", "Switzerland", "Italy",
    "England", "Austria", "France", "Netherlands", "Denmark",
    "Slovenia", "Romania", "Belgium", "Slovakia", "Turkey", "Georgia"
]

filtered_data = final_df[
    ((final_df['Year'] == 1996) & (final_df['Country'] == 'Germany')) |
    ((final_df['Year'] == 2000) & (final_df['Country'] == 'France')) |
    ((final_df['Year'] == 2004) & (final_df['Country'] == 'Greece')) |
    ((final_df['Year'] == 2008) & (final_df['Country'] == 'Spain')) |
    ((final_df['Year'] == 2012) & (final_df['Country'] == 'Spain')) |
    ((final_df['Year'] == 2016) & (final_df['Country'] == 'Portugal')) |
    ((final_df['Year'] == 2020) & (final_df['Country'] == 'Italy'))
]

age_stats_filtered = filtered_data.groupby(['Year', 'Country'])['Age'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(14, 10))

colors_2024 = ['red' if country in highlight_countries else 'gray' if country not in age_stats_filtered['Country'].unique() else 'blue' for country in age_stats_2024['Country']]
plt.scatter(age_stats_2024['mean'], age_stats_2024['std'], c=colors_2024, alpha=0.7)

for i, row in age_stats_2024.iterrows():
    color = 'red' if row['Country'] in highlight_countries else 'gray' if row['Country'] not in age_stats_filtered['Country'].unique() else 'blue'
    plt.text(row['mean'], row['std'], f"{row['Country']}", fontsize=12, color=color)

plt.scatter(age_stats_filtered['mean'], age_stats_filtered['std'], marker='o', color='blue')

for i, row in age_stats_filtered.iterrows():
    plt.text(row['mean'], row['std'], f"{row['Country']}\n({row['Year']})", fontsize=12, ha='center', va='bottom', color='blue')

plt.title('Average Age and Standard Deviation: Euro 2024 Teams, Round of 16, and Champions from 1996 to 2020')
plt.xlabel('Average', fontsize=14)
plt.ylabel('Standard Deviation', fontsize=14)
plt.grid(True)

legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Round of 16 (2024)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Eliminated (2024)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Champions (1996-2020)')
]
plt.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()