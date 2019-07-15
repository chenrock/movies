import pandas as pd


# ----- read in movies csv table as df
movies_df = pd.read_csv('data/db-csv-table_movies.csv').drop(columns=['index', 'id'])
# ----- read in extra_omdb_data csv table as df
extra_omdb_data_df = pd.read_csv('data/db-csv-table_extra_omdb_data.csv').drop(columns=['index'])


# ----- extracting IMDb ID from 'movie_imdb_link' column in movies_df
movie_imdb_link_list = movies_df['movie_imdb_link'].to_list()
imdb_id_list = []
for link in movie_imdb_link_list:
    imdb_id_list.append(link.split('/')[-2])

# ----- add IMDb ID column to movies_df
movies_df['imdbID'] = imdb_id_list

# ----- join movies_df and extra_omdb_data_df
final_df = pd.merge(movies_df, extra_omdb_data_df, on='imdbID', how='left')


# ----- get columns in final_df
final_df_columns = final_df.columns.tolist()
# ----- exploratory data analysis
final_df['Title'].nunique()
# 4,639 so all unique

final_df['movie_facebook_likes'].describe()
final_vc_movie_fb_likes = final_df[['movie_title', 'movie_facebook_likes']].sort_values(['movie_facebook_likes'],
                                                                                        ascending=False)

final_vc_color = final_df['color'].value_counts().sort_index()
# Black and White 193; Color 4,431; Missing 15

final_vc_director_name = final_df['director_name'].value_counts().sort_values(ascending=False)

final_vc_director_fb_likes = final_df[['director_name', 'director_facebook_likes']].\
    groupby(['director_name', 'director_facebook_likes']).count().reset_index().\
    sort_values(['director_facebook_likes'], ascending=False)

final_vc_actor_1_name = final_df['actor_1_name'].value_counts().sort_values(ascending=False)
final_vc_actor_2_name = final_df['actor_2_name'].value_counts().sort_values(ascending=False)
final_vc_actor_3_name = final_df['actor_3_name'].value_counts().sort_values(ascending=False)

final_vc_actor_1_fb_likes = final_df[['actor_1_name', 'actor_1_facebook_likes']].\
    groupby(['actor_1_name', 'actor_1_facebook_likes']).count().reset_index().\
    sort_values(['actor_1_facebook_likes'], ascending=False)

final_vc_actor_2_fb_likes = final_df[['actor_2_name', 'actor_2_facebook_likes']].\
    groupby(['actor_2_name', 'actor_2_facebook_likes']).count().reset_index().\
    sort_values(['actor_2_facebook_likes'], ascending=False)

final_vc_actor_3_fb_likes = final_df[['actor_3_name', 'actor_3_facebook_likes']].\
    groupby(['actor_3_name', 'actor_3_facebook_likes']).count().reset_index().\
    sort_values(['actor_3_facebook_likes'], ascending=False)


final_vc_genres = final_df['genres'].value_counts().sort_values(ascending=False)

final_df['budget'].describe()
# foreign exchange currency rates required in budget

final_df['gross'].describe()

final_vc_content_rating = final_df['content_rating'].value_counts().sort_values(ascending=False)

# popular
final_df['imdb_score'].describe()

final_vc_writer = final_df['Writer'].value_counts().sort_values(ascending=False)

final_vc_awards = final_df['Awards'].value_counts().sort_values(ascending=False)

# critics
final_df['Metascore'].describe()

final_vc_type = final_df['Type'].value_counts().sort_values(ascending=False)

print(final_df.shape)

final_df_dropna = final_df.dropna()
print(final_df_dropna.shape)
