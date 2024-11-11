import nltk, random
import bs4 as bs
from bs4 import BeautifulSoup as bsoup
from urllib import request
import sqlite3
import json
import time
from nltk import *
from database import *
from qanda import *
from smalltalk import *

class Chatbot:
    def __init__(self, database):
        self.name = database
        self.sentiment = 0
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger_eng')
        build_dt_matrix(questions)
        smalltalk_init('smalltalk')
    
    def main(self):
        database = Database(self.name)
        database.connect()
        prompts = ["How are you today?","What's the weather like today?","What's the time?","What can you do for me?",
                   "What's your favourite food?","Recommend things to do near me."]
        userInput = None
        try:
            #check if user  1 exists
            database.sqlexecute('''SELECT * FROM UserInfo WHERE userID=1''')
            user = database.cursor.fetchone()

            if user:
                #greet user
                print(f"\nTo exit the chat, type 'exit'\n")
                print(f"<{self.name}>: Hi {user[1]}!")
            else:
                #ask for name
                print(f"<{self.name}>: Hi! My name is {self.name}.\n What's your name?")
                userInput = input("<you>: ")
                #insert username into database
                database.sqlexecute(f'''INSERT INTO UserInfo (name) VALUES ('{userInput}')''')
                print(f"<{self.name}>: Hi {userInput}!\n To exit the chat, type 'exit'.")
                database.sqlexecute('''SELECT * FROM UserInfo WHERE userID=1''')
                user = database.cursor.fetchone()

            while True:
                #main loop
                print(f"<{self.name}>: How can I help you today?")
                print(f"\033[3m\033[34mNot sure what to ask?\nTry:\n{random.choice(prompts)}\033[0m")
                userInput = input(f"<{user[1]}>: ").lower()

                if userInput == 'exit':
                    print(f"<{self.name}>: Goodbye!")
                    break

                output = find_response(userInput)
                while (not output) & (userInput!='exit'):
                    #smalltalk loop
                    print(f"<{self.name}>: I didn't understand that, can you say it again?")
                    userInput = input(f"<{user[1]}>: ")
                    output = find_response(userInput)
                if output:
                    print(f"<{self.name}>: {output}")

        finally:
            database.connection.close()