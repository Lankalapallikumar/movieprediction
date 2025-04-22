import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('IMDb Movies India.csv', encoding='latin1')
df = df.dropna(subset=['Rating'])
df['Votes'] = df['Votes'].str.replace(',', '').astype(float)
df['Duration'] = df['Duration'].str.extract(r'(\d+)').astype(float)
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(float)

director_avg = df.groupby('Director')['Rating'].mean().to_dict()
df['Director_Success'] = df['Director'].map(director_avg)

genre_avg = (
    df
    .assign(Genre=df['Genre'].str.split(r',\s*'))
    .explode('Genre')
    .groupby('Genre')['Rating']
    .mean()
    .to_dict()
)

df['Genre'] = df['Genre'].fillna('')

genre_freq = (
    df
    .assign(Genre=df['Genre'].str.split(r',\s*'))
    .explode('Genre')
    .Genre
    .value_counts()
    .to_dict()
)

def avg_genre_rating(g):
    if pd.isna(g): return np.nan
    lst = [x.strip() for x in g.split(',')]
    vals = [genre_avg.get(x, np.nan) for x in lst]
    return np.nanmean(vals)

df['Genre_Avg_Rating'] = df['Genre'].apply(avg_genre_rating)

actor1_avg = df.groupby('Actor 1')['Rating'].mean().to_dict()
df['Actor1_Success'] = df['Actor 1'].map(actor1_avg)

df['Genre_Count'] = df['Genre'].fillna('').apply(lambda x: len(x.split(',')))

features = [
    'Duration', 'Votes', 'Director_Success', 'Genre_Avg_Rating', 
    'Actor1_Success', 'Genre_Count'
]
X = df[features]
y = df['Rating']

num_pipeline = Pipeline([('impute', SimpleImputer(strategy='mean'))])
preprocessor = ColumnTransformer([('num', num_pipeline, features)])

model = Pipeline([
    ('prep', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score:            {r2_score(y_test, y_pred):.2f}")

print("\n--- Predict a New Movie’s IMDb Rating ---")
year = float(input("Release Year (e.g. 2023): ").strip())
duration = float(input("Duration in minutes: ").strip())
votes = float(input("Expected votes (e.g. 15000): ").strip())
director = input("Director name: ").strip()
genres = input("Genre(s), comma-separated: ").strip()
actor1 = input("Lead actor name: ").strip()

dir_succ = director_avg.get(director, df['Rating'].mean())
gen_succ = avg_genre_rating(genres)
act1_succ = actor1_avg.get(actor1, df['Rating'].mean())
genre_list = [g.strip() for g in genres.split(',') if g.strip()]
genre_cnt = len(genre_list)

new_row = pd.DataFrame([{
    'Duration': duration,
    'Votes': votes,
    'Director_Success': dir_succ,
    'Genre_Avg_Rating': gen_succ,
    'Actor1_Success': act1_succ,
    'Genre_Count': genre_cnt
}])

pred = model.predict(new_row)[0]
print(f"\nPredicted IMDb Rating: {pred:.2f}")

rf_model = model.named_steps['rf']
importances = rf_model.feature_importances_
feat_importance = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feat_importance.plot(kind='barh', color='skyblue')
plt.title('Feature Importance in Rating Prediction')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(df['Rating'], bins=20, kde=True, color='lightgreen')
plt.axvline(pred, color='red', linestyle='--', label=f'Predicted Rating: {pred:.2f}')
plt.title('Predicted Rating vs IMDb Rating Distribution')
plt.xlabel('IMDb Rating')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.show()

popularity = [genre_freq.get(g, 0) for g in genre_list]
ratings = [genre_avg.get(g, np.nan) for g in genre_list]

ratings = [r if not np.isnan(r) else 0 for r in ratings]

plt.figure(figsize=(7, 5))
sns.scatterplot(x=popularity, y=ratings, s=100)
for i, g in enumerate(genre_list):
    plt.text(popularity[i] + 1, ratings[i], g, fontsize=9)
plt.xlabel('Genre Popularity (# of Movies)')
plt.ylabel('Average Rating')
plt.title('Genre Popularity vs Avg Rating (Your Genres)')
plt.tight_layout()
plt.show()
