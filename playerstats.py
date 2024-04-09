import pandas as pd

# remove categorical columns like team, position
player_stats_df = pd.read_csv('https://query.data.world/s/ksxh26pj3drsht2otdqfe2iwbjpt6f?dws=00000')
columns_to_keep = ['Player', 'Age', 'G', 'GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'Year']
player_stats_correct_columns = player_stats_df[columns_to_keep]

# all we need is a dataframe like this with the all star information from the all-stars
# can change year format to be single year if we need
test_example_all_star_df = pd.DataFrame({'Player': ["LeBron James", "Stephen Curry", "Malik Monk"],
                                 'Year': ['2006-2007', '2017-2018', '2020-2021'],
                                 'All-Star': [1, 1, 0]})

# the big join
test_example_data = pd.merge(player_stats_correct_columns, test_example_all_star_df,
                             how='left', left_on=['Player', 'Year'], right_on=['Player', 'Year'])

# all star column needs to be 1s and 0s, fillna with 0
test_example_data["All-Star"] = test_example_data['All-Star'].fillna(0)

# at this point we should remove player name from the data
# and should also remove 'honorary' all-star selections that were based on
# name recognition like Dwayne Wade's final year and Yao Ming in 2011
# also change year to a numerical value since the start of the dataset? so its from 0-22 ish?

# todo here we standardize data so the appropriate columns have mean=0 std=1

# test_result = test_example_data
# test_result = test_example_data[test_example_data['All-Star'] == 1]
test_result = test_example_data[test_example_data['Player'] == "Stephen Curry"]
print(test_result)

