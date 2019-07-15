import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix


########################################################################################################################
# ----- section for defining functions
########################################################################################################################


# ----- function to help create new row in df which will signify above or below average critic rating
def label_critic(row):
    if row['Metascore'] >= 55.601932:
        return 'Above Average'
    else:
        return 'Below Average'


# ----- function to help create new row in df which will signify above or below average popular rating
def label_popular(row):
    if row['imdb_score'] >= 6.542971:
        return 'Above Average'
    else:
        return 'Below Average'


# ----- function to help create new row in df which will signify above or below average social media buzz
def label_buzz(row):
    if row['movie_facebook_likes'] >= 3486258.15856:
        return 'Above Average'
    else:
        return 'Below Average'


########################################################################################################################
# ----- this section reads in the tables from the database as dfs then joins the df together
########################################################################################################################

# ----- connect to db
db = sqlite3.connect('data/movies.db')


# ----- read in movies db table as df
movies_df = pd.read_sql('select * from movies', db).drop(columns=['id'])
# ----- read in extra_omdb_data db table as df
extra_omdb_data_df = pd.read_sql('select * from extra_omdb_data', db)

# ---- close connection to db
db.close()


# ----- extracting IMDb ID from 'movie_imdb_link' column in movies_df to create a column to join to extra_omdb_data_df
movie_imdb_link_list = movies_df['movie_imdb_link'].to_list()
imdb_id_list = []
for link in movie_imdb_link_list:
    imdb_id_list.append(link.split('/')[-2])
# ----- add IMDb ID column to movies_df
movies_df['imdbID'] = imdb_id_list


# ----- join movies_df and extra_omdb_data_df
final_df = pd.merge(movies_df, extra_omdb_data_df, on='imdbID', how='left')


########################################################################################################################
# ----- this section cleans the final_df created above and applies some feature engineering techniques
########################################################################################################################

# ----- create list of items to replace with np.nan using knowledge gained from exploring data in eda.py program
to_replace_list = ['', 'null', 'Null', 'NULL', 'missing', 'Missing', 'MISSING', 'na', 'NA', 'n/a', 'N/A', '  ?']
replace_with_list = [np.nan]*len(to_replace_list)

# ----- remove missing values
final_df_dropna = final_df.replace(to_replace_list, replace_with_list).dropna().reset_index(drop=True)
print(final_df_dropna.shape)
print('final_df shape:', final_df.shape, '\n',
      'final_df_dropna shape:', final_df_dropna.shape, '\n',
      'rows dropped:', (final_df.shape[0] - final_df_dropna.shape[0]))

# ----- check data type of columns
print(final_df_dropna.dtypes)
# all objects, so need to convert numbers to float64

# ----- columns names that should be float64 instead of object data types
columns_float64 = ['movie_facebook_likes', 'prolific_director', 'director_facebook_likes', 'prolific_actor_1',
                   'prolific_actor_2', 'prolific_actor_3', 'actor_1_facebook_likes', 'actor_2_facebook_likes',
                   'actor_3_facebook_likes', 'budget', 'gross', 'num_critic_for_reviews', 'num_voted_users',
                   'facenumber_in_poster', 'num_user_for_reviews', 'title_year', 'imdb_score', 'oscar_nom_movie',
                   'oscar_nom_actor', 'duration', 'aspect_ratio', 'Metascore']

# ----- for loop to convert columns in columns_float64 list to float64 dtype
for column in columns_float64:
    final_df_dropna[column] = final_df_dropna[column].astype('float64')

# ----- get summary statistics for the variable that is an approximate measure of critical acclaim
print(final_df_dropna['Metascore'].describe())
# mean: 55.601932; std: 18.150903

# ----- get summary statistics for the variable that is an approximate measure of popular acclaim
print(final_df_dropna['imdb_score'].describe())
# mean: 6.542971; std: 1.042431

# ----- get summary statistics for the variables that is an approximate measure of social media buzz
print(final_df_dropna['movie_facebook_likes'].describe())
# mean: 3,486,258.15856; std: 11,958,892.91907
print(final_df_dropna['director_facebook_likes'].describe())
# mean: 895.024983; std: 3,273.888857
print(final_df_dropna['actor_1_facebook_likes'].describe())
# mean: 3,484,254.80413; std: 11,959,472.16995
print(final_df_dropna['actor_2_facebook_likes'].describe())
# mean: 2,038.63491; std: 4,642.74485
print(final_df_dropna['actor_3_facebook_likes'].describe())
# mean: 781.47435; std: 1,915.68800


# ----- apply functions to create new columns
final_df_dropna['Critical_Acclaim'] = final_df_dropna.apply(lambda row: label_critic(row), axis=1)
final_df_dropna['Popular_Acclaim'] = final_df_dropna.apply(lambda row: label_popular(row), axis=1)
final_df_dropna['Social_Media_Buzz'] = final_df_dropna.apply(lambda row: label_buzz(row), axis=1)

# ----- create target variable
final_df_dropna['Target'] = final_df_dropna['Critical_Acclaim'] + '_' +\
                            final_df_dropna['Popular_Acclaim'] + '_' +\
                            final_df_dropna['Social_Media_Buzz']

# ----- check distribution of target variable
final_df_dropna['Target'].value_counts().sort_values(ascending=False)


########################################################################################################################
# ----- this section prepares the data for training and then trains the model
########################################################################################################################

# ---- create x and y df
x = final_df_dropna[['color', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'budget', 'genres',
                     'language', 'country', 'content_rating', 'duration', 'Writer']]
y = final_df_dropna.loc[:, final_df_dropna.columns == 'Target']

# ----- split train and test
x_train_imbal, x_test, y_train_imbal, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# ----- join x_train and y_train to apply manual oversampling to combat target class imbalance
train_imbal = pd.concat([x_train_imbal, y_train_imbal], axis=1, sort=False)
train_imbal['Target'].value_counts().sort_values(ascending=False)
# Above Average_Above Average_Below Average    754
# Below Average_Below Average_Below Average    661
# Below Average_Above Average_Below Average    227
# Above Average_Below Average_Below Average    160
# Above Average_Above Average_Above Average    115
# Below Average_Below Average_Above Average    115
# Below Average_Above Average_Above Average     48
# Above Average_Below Average_Above Average     21


# ----- manually resample to create balanced training data set
train_AA_AA_BA = train_imbal.loc[train_imbal['Target'] == 'Above Average_Above Average_Below Average']
train_BA_BA_BA = train_imbal.loc[train_imbal['Target'] == 'Below Average_Below Average_Below Average'].\
    sample(754, replace=True, random_state=0)
train_BA_AA_BA = train_imbal.loc[train_imbal['Target'] == 'Below Average_Above Average_Below Average'].\
    sample(754, replace=True, random_state=0)
train_AA_BA_BA = train_imbal.loc[train_imbal['Target'] == 'Above Average_Below Average_Below Average'].\
    sample(754, replace=True, random_state=0)
train_AA_AA_AA = train_imbal.loc[train_imbal['Target'] == 'Above Average_Above Average_Above Average'].\
    sample(754, replace=True, random_state=0)
train_BA_BA_AA = train_imbal.loc[train_imbal['Target'] == 'Below Average_Below Average_Above Average'].\
    sample(754, replace=True, random_state=0)
train_BA_AA_AA = train_imbal.loc[train_imbal['Target'] == 'Below Average_Above Average_Above Average'].\
    sample(754, replace=True, random_state=0)
train_AA_BA_AA = train_imbal.loc[train_imbal['Target'] == 'Above Average_Below Average_Above Average'].\
    sample(754, replace=True, random_state=0)

# ----- concatenate manally oversampled df to create balanced training df
train_bal = pd.concat([train_AA_AA_BA,
                       train_BA_BA_BA,
                       train_BA_AA_BA,
                       train_AA_BA_BA,
                       train_AA_AA_AA,
                       train_BA_BA_AA,
                       train_BA_AA_AA,
                       train_AA_BA_AA]).reset_index(drop=True)

# ----- split train_bal into x and y df
x_train_bal = train_bal.loc[:, train_bal.columns != 'Target']
y_train_bal = train_bal.loc[:, train_bal.columns == 'Target']['Target'].values


# ----- encode categorical variables
preprocess = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler()), ['budget', 'duration']),
    (OneHotEncoder(handle_unknown='ignore'), ['color', 'director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name',
                                              'genres', 'language', 'country', 'content_rating', 'Writer']))

logreg = LogisticRegression(class_weight='balanced', multi_class='ovr', solver='liblinear')


# ----- fit logistic regression
model = make_pipeline(
    preprocess,
    logreg)
model.fit(x_train_bal, y_train_bal)


# ----- feature coefficients
feature_coef = np.transpose(model[1].coef_)


# ----- predict on x_test
y_pred = model.predict(x_test)

# ----- predict on x_train_imbal
y_pred_train_imbal = model.predict(x_train_imbal)

# ----- predict on x_train_bal
y_pred_train_bal = model.predict(x_train_bal)

# ----- predict on x_all
y_pred_all = model.predict(final_df_dropna[[]])

print('Accuracy of logistic regression classifier on test set: {:.5f}'.format(model.score(x_test, y_test)))
# Accuracy of logistic regression classifier on test set: 0.51720

print('Accuracy of logistic regression classifier on train set: {:.5f}'.format(model.score(x_train_imbal,
                                                                                           y_train_imbal)))
# Accuracy of logistic regression classifier on imbalanced train set: 0.96002
# overfitting

print('Accuracy of logistic regression classifier on train set: {:.5f}'.format(model.score(x_train_bal,
                                                                                           y_train_bal)))
# Accuracy of logistic regression classifier on oversampled balanced train set: 0.99884
# overfitting


# ----- confusion matrix for y_pred on x_test
cnfsn_mtrx = confusion_matrix(y_test, y_pred)
print(cnfsn_mtrx)

# ----- create df of y_test and y_pred which will be combined with all column data below
y_test_pred = pd.DataFrame(
    {'y_test': list(y_test['Target']),
     'y_pred': y_pred
    })

# ----- getting all columns to concat with y_test_pred
x_test_index = x_test.reset_index(drop=False)['index'].tolist()
x_test_all_columns = final_df_dropna.loc[x_test_index].reset_index(drop=True)

# ----- join x_test_all_columns and y_test_pred to export to Excel
x_test_all_columns_with_pred = pd.concat([x_test_all_columns, y_test_pred], axis=1, sort=False)

# ----- export x_test_all_columns_with_pred to Excel
x_test_all_columns_with_pred.to_excel('data/x_test_all_columns_with_pred.xlsx')


########################################################################################################################
# ----- this section creates df with predictions and then connects
########################################################################################################################

# ----- convert y_pred_all to df and name column
y_pred_all_df = pd.DataFrame(y_pred_all)
y_pred_all_df = y_pred_all_df.rename(columns={y_pred_all_df.columns[0]: "Predicted"})

# ----- create df will predictions for all rows that were not dropped
all_predictions = pd.concat([final_df_dropna, y_pred_all_df], axis=1, sort=False)

# ----- connect to db
db = sqlite3.connect('data/movies.db')

# ----- write records stored in df to db
all_predictions.to_sql('prediction_data', db, if_exists='replace', index=False)

# ----- close db
db.close()
