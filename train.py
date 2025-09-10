#import the dependencies 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
#load the dataset
df = pd.read_csv("spam.csv",encoding='latin-1')


#choose the  dataset which are useful
df = df[['v1','v2']]
df.head()


#as the data columns is not appropriate we will change the column name 
df.columns=['label','message']
df.head()


#now lets check whether any column has the missing values
df.isnull().sum()
#so there is no null values we proceed to next


#now encode the label and message as the model cant take 
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])
df.head()  #well both of the 


#now split the column for train and test 
X = df['message']
Y = df['label']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42,stratify=Y)



#now build the model
model = make_pipeline(
    TfidfVectorizer(stop_words="english", ngram_range=(1,2)),
    MultinomialNB()
)
#now fit the model
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
Y_pred
#check the accuracy
accuracy_score = classification_report(Y_test,Y_pred)
print(accuracy_score)
joblib.dump(model,"sentiment_pipeline.pkl")
