import json


file_path = '../data/smalltalk.json'
# Load data from the JSON file
def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Save data to the JSON file
def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)

# Add a new intent
def add_intent(intent_name, examples, file_path):
    data = load_data(file_path)
    data['intent'].append({"tag": intent_name, "patterns": examples})
    save_data(data, file_path)

# Retrieve an intent by name
def get_intent(intent_name, file_path):
    data = load_data(file_path)
    intent = next((intent for intent in data['intent'] if intent['tag'] == intent_name), None)
    return intent

# Update an existing intent
def update_intent(intent_name, new_examples, file_path):
    data = load_data(file_path)
    for intent in data['intent']:
        if intent['tag'] == intent_name:
            intent['patterns'] = new_examples
            save_data(data, file_path)
            return
    add_intent(intent_name, new_examples, file_path)

# Delete an intent
def delete_intent(intent_name, file_path):
    data = load_data(file_path)
    data['intent'] = [intent for intent in data['intent'] if intent['tag'] != intent_name]
    save_data(data, file_path)

#prepare embeddings
def create_embedding(smalltalk, model):
    intent_embeddings = {}
    for intent in smalltalk['intent']:
        patterns = intent['patterns']
        embeddings = model.encode(patterns, convert_to_tensor=True)
        intent_embeddings[intent['tag']] = embeddings
    return intent_embeddings

def best_match(input, intent_embeddings, model):
    input_embedding = model.encode(input, convert_to_tensor=True)
    match = None
    score = -1
    for tag, embeddings in intent_embeddings.items():
        scores= util.cos_sim(input_embedding, embeddings)[0] #cosine similarity
