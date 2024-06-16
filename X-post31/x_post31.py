import pandas as pd
import numpy as np

results = pd.read_csv('results.csv')
euro_2024_countries = pd.read_excel('euro_2024_countries.xlsx')
matches = pd.read_excel('matches.xlsx')

results['date'] = pd.to_datetime(results['date'])
results = results.dropna(subset=['home_score','away_score'])

results['home_team'] = results['home_team'].replace('Czech Republic', 'Czechia')
results['away_team'] = results['away_team'].replace('Czech Republic', 'Czechia')

results = results[
    (results['home_team'].isin(euro_2024_countries['country'])) &
    (results['away_team'].isin(euro_2024_countries['country']))
]

results = results[results['tournament'].isin(['UEFA Euro','UEFA Euro qualification','UEFA Nations League','FIFA World Cup','FIFA World Cup qualification'])]

results['home_score'] = pd.to_numeric(results['home_score'], errors='coerce').astype(int)
results['away_score'] = pd.to_numeric(results['away_score'], errors='coerce').astype(int)

home_team_stats = results.groupby('home_team').agg({
    'home_score': ['sum', 'count']
}).reset_index()
home_team_stats.columns = ['home_team', 'total_goals', 'matches_played']

away_team_stats = results.groupby('away_team').agg({
    'away_score': ['sum', 'count']
}).reset_index()
away_team_stats.columns = ['away_team', 'total_goals', 'matches_played']

team_stats = pd.merge(home_team_stats, away_team_stats, left_on='home_team', right_on='away_team', how='outer')
team_stats['total_goals'] = team_stats['total_goals_x'] + team_stats['total_goals_y']
team_stats['total_matches'] = team_stats['matches_played_x'] + team_stats['matches_played_y']
team_stats = team_stats[['home_team','total_goals','total_matches']]

team_stats['overall_lambda'] = team_stats['total_goals'] / team_stats['total_matches']

def simulate_match(team1, team2, team_stats, num_simulations=100000):
    team1_avg_goals = team_stats.loc[team_stats['home_team'] == team1, 'overall_lambda'].values[0]
    team2_avg_goals = team_stats.loc[team_stats['home_team'] == team2, 'overall_lambda'].values[0]

    team1_goals_simulated = np.random.poisson(team1_avg_goals, num_simulations)
    team2_goals_simulated = np.random.poisson(team2_avg_goals, num_simulations)

    team1_total_wins = np.sum(team1_goals_simulated > team2_goals_simulated)
    team2_total_wins = np.sum(team2_goals_simulated > team1_goals_simulated)
    draws = np.sum(team1_goals_simulated == team2_goals_simulated)

    if team1_total_wins > team2_total_wins:
        match_result = f'{team1} wins'
    elif team2_total_wins > team1_total_wins:
        match_result = f'{team2} wins'
    else:
        match_result = 'No winner'

    return {
        'Team1 Wins Prob': team1_total_wins / num_simulations,
        'Team2 Wins Prob': team2_total_wins / num_simulations,
        'Draws Prob': draws / num_simulations,
        'Match Result': match_result
    }

simulations = []
for index, row in matches.iterrows():
    team1 = row['team1']
    team2 = row['team2']

    match_simulations = simulate_match(team1, team2, team_stats)

    simulations.append(match_simulations)

simulations_df = pd.DataFrame(simulations)

matches_with_simulations = pd.concat([matches, simulations_df], axis=1)

matches_with_simulations = matches_with_simulations.rename(columns={'team1':'Team1', 'team2':'Team2'})

def highlight_max_prob(row):
    max_prob = row[['Team1 Wins Prob', 'Team2 Wins Prob', 'Draws Prob']].max()
    return ['background-color: red' if v == max_prob else '' for v in row]

styled_matches = matches_with_simulations.style.apply(highlight_max_prob, axis=1)

html_styled_matches = styled_matches.to_html()

with open('styled_matches.html', 'w') as f:
    f.write(html_styled_matches)