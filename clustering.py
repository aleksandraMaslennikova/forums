from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import readData

listDocuments = readData.readDataFromXML("data/final_users_location_user_id.txt")

documentsBagOfWords = []
for document in listDocuments:
    documentsBagOfWords.append(readData.documentInBagOfLemmas(document["document"]))

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(documentsBagOfWords)

kmeans = KMeans(n_clusters=2).fit(tfidf)
print(kmeans)