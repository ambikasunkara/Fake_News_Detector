import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)   # NEW FIX
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text_basic(text):
    # Lowercase
    text = text.lower()

    # Remove non-alphabetic characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Remove stopwords + Lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return " ".join(tokens)
