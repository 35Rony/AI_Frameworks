import spacy
from spacy.matcher import Matcher

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Example review text (replace with your dataset)
text = """
I recently bought the Apple iPhone 12 and I love it! The Samsung Galaxy S21 is also great, but I prefer the iPhone.
Terrible experience with the Sony headphones, very disappointed.
"""

# Process text with spaCy NLP pipeline
doc = nlp(text)

# Define custom patterns to extract product names and brands (besides built-in NER)
matcher = Matcher(nlp.vocab)
patterns = [
    [{"ENT_TYPE": "ORG"}],  # Brand names
    [{"ENT_TYPE": "PRODUCT"}],  # Product names
    [{"LOWER": "iphone"}],
    [{"LOWER": "samsung"}],
    [{"LOWER": "sony"}]
]
matcher.add("BRAND_PRODUCT", patterns)

matches = matcher(doc)
entities = set()
for match_id, start, end in matches:
    span = doc[start:end]
    entities.add(span.text)

# Extract named entities from built-in NER as well
for ent in doc.ents:
    if ent.label_ in ("ORG", "PRODUCT"):
        entities.add(ent.text)

print("Extracted Entities (Brands and Products):")
print(entities)

# Simple rule-based sentiment analysis keywords
positive_words = {"love", "great", "excellent", "amazing", "good", "perfect"}
negative_words = {"terrible", "disappointed", "bad", "poor", "worst"}

# Basic sentiment scoring
pos_score = sum(token.text.lower() in positive_words for token in doc)
neg_score = sum(token.text.lower() in negative_words for token in doc)

sentiment = "Neutral"
if pos_score > neg_score:
    sentiment = "Positive"
elif neg_score > pos_score:
    sentiment = "Negative"

print("Sentiment:", sentiment)
