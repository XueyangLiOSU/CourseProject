import json
import torch
import random
from nltk_utils import tokenize, bag_of_words
from model import NeuralNet
from Predictor_py_version import sentiment_analysis

# setting up files, parameters and learning model
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = 'data.pth'
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# give the chat bot a proper name
bot_name = 'Eva'

"""
main function for chat
msg: input sentence from the GUI interface
return: a response sentence to the GUI interface
"""
def get_response(msg):
    # set the default response
    response = "Sorry that I don't seem to understand what you just said..."    

    # apply the sentiment analysis first
    sentiment_tag = sentiment_analysis(msg)
    if sentiment_tag == -1:
        tag = 'negative'
        for i in intents['intents']:
            if tag == i['tag']:
                response = random.choice(i['responses'])
                return response
    elif sentiment_tag == 1:
        tag = 'positive'
        for i in intents['intents']:
            if tag == i['tag']:
                response = random.choice(i['responses'])
                return response
    else:
        # if neutral, do the usual chatbot routine
        sentence = tokenize(msg)
        x = bag_of_words(sentence, all_words)
        x = x.reshape(1, x.shape[0])
        x = torch.from_numpy(x)

        output = model(x)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.8:
            for i in intents['intents']:
                if tag == i['tag']:
                    response = random.choice(i['responses'])
                    return response

    return response

""" # chat loop for command line interface, we do not need this since we now have GUI
print("Eva is connected! (Type 'exit' to disconnect)")

while True:
    sentence = input('You: ')
    if sentence == 'exit':
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.8:
        for i in intents['intents']:
            if tag == i ['tag']:
                print(f"{bot_name}: {random.choice(i['responses'])}")
    else:
        print(f"{bot_name}: Sorry that I don't seem to understand what you just said...")
"""
    