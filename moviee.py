import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('IMDb Movies India.csv', encoding='ISO-8859-1')
df.columns = df.columns.str.strip()
df = df[df['Rating'].notnull()]
df['Duration'] = df['Duration'].str.extract(r'(\d+)')
df['Duration'] = pd.to_numeric(df['Duration'], errors='coerce')
df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
df['Extracted_Year'] = df['Name'].str.extract(r'\((\d{4})\)').astype(float)
df['Year'] = df['Year'].fillna(df['Extracted_Year'])
df.drop(columns=['Extracted_Year'], inplace=True)
df.dropna(subset=['Year', 'Duration', 'Genre', 'Votes', 'Director'], inplace=True)

director_avg_rating = df.groupby("Director")["Rating"].mean().to_dict()
df["Director_Avg_Rating"] = df["Director"].map(director_avg_rating)
df['Main_Genre'] = df['Genre'].str.split(',').str[0]
genre_avg_rating = df.groupby('Main_Genre')['Rating'].mean().to_dict()
df['Genre_Avg_Rating'] = df['Main_Genre'].map(genre_avg_rating)
actor_avg_rating = df.groupby("Actor 1")["Rating"].mean().to_dict()
df["Actor1_Avg_Rating"] = df["Actor 1"].map(actor_avg_rating)

le_director = LabelEncoder()
le_genre = LabelEncoder()
df["Director_Encoded"] = le_director.fit_transform(df["Director"])
df["Genre_Encoded"] = le_genre.fit_transform(df["Main_Genre"])

features = ['Duration', 'Votes', 'Director_Avg_Rating', 'Genre_Avg_Rating', 'Actor1_Avg_Rating', 'Director_Encoded', 'Genre_Encoded']
X = df[features]
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"Model RMSE: {rmse:.2f}")

duration = float(input("Enter movie duration in minutes: "))
votes = int(input("Enter number of votes: "))
director = input("Enter director name: ")
genre = input("Enter main genre: ")
actor1 = input("Enter lead actor name: ")

dir_avg = director_avg_rating.get(director, df['Director_Avg_Rating'].mean())
gen_avg = genre_avg_rating.get(genre, df['Genre_Avg_Rating'].mean())
act_avg = actor_avg_rating.get(actor1, df['Actor1_Avg_Rating'].mean())
dir_enc = le_director.transform([director])[0] if director in le_director.classes_ else 0
gen_enc = le_genre.transform([genre])[0] if genre in le_genre.classes_ else 0

input_df = pd.DataFrame([{
    'Duration': duration,
    'Votes': votes,
    'Director_Avg_Rating': dir_avg,
    'Genre_Avg_Rating': gen_avg,
    'Actor1_Avg_Rating': act_avg,
    'Director_Encoded': dir_enc,
    'Genre_Encoded': gen_enc
}])

predicted_rating = model.predict(input_df)[0]
print(f"Predicted IMDb Rating: {predicted_rating:.2f}")
