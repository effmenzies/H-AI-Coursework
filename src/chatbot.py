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
from user_input import *

class Chatbot:
    def __init__(self, database):
        self.name = database
        self.sentiment = 0
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('universal_target')
        build_dt_matrix(questions)
        smalltalk_init('smalltalk')
    
    def main(self):
        database = Database(self.name)
        database.connect()
        prompts = ["How are you today?","What's the weather like today?","What's the time?","What can you do for me?",
                   "What's your favourite food?","Recommend things to do near me."]
        try:
            #check if user  1 exists
            database.sqlexecute('''SELECT * FROM UserInfo WHERE userID=1''')
            user = database.cursor.fetchone()
            userName=None
            userInput=None
            print(f"<{self.name}>: Hi! My name is {self.name}.")
            if user:
                #greet user
                userName = user[1]
                print(f"\nTo exit the chat, type 'exit'\n")
                print(f"<{self.name}>: Hi {userName}!")
            else:
                while not userName:
                    #ask for name
                    print(f"What's your name?")
                    userInput = input("<you>: ")
                    names = extract_names(userInput)
                    fullName = max(names)
                    nickName = None
                    print(f"<{self.name}>: Hi, I've got your full name as {fullName}.")
                    if len(names)>1:
                        nickName = min(names)
                        print(f"And {nickName} as a nickname.")
                    print(f"Is this correct?")
                    userInput = input("<you>: ")
                    if match_output(userInput, 'smalltalk') in ('confirmation_yes','agent_right'):
                        while not nickName:
                            print(f"<{self.name}>: Is {fullName} your preferred name?")
                            userInput = input("<you>: ")
                            if match_output(userInput, 'smalltalk') in ('confirmation_yes','agent_right'):
                                nickName=fullName
                            elif match_output(userInput, 'smalltalk') in ('confirmation_no','agent_wrong'):
                                print(f"<{self.name}>: What is your prefered name?")
                                userInput = input("<you>: ")
                                nickName = extract_names(userInput)[0]
                                print(f"<{self.name}>: I'll call you {nickName}.")
                            else: print(f"<{self.name}>: Sorry, I misunderstood you.")

                        #insert username into database
                        database.sqlexecute(f'''INSERT INTO UserInfo (name, nickname) VALUES ('{fullName}','{nickName}')''')
                        database.sqlexecute('''SELECT * FROM UserInfo WHERE userID=1''')
                        user = database.cursor.fetchone()
                        userName = nickName
                        print(f"<{self.name}>: Hi {userName}!\n To exit the chat, type 'exit'.")
                    else:
                        print(f"<{self.name}>: Sorry, I misunderstood.")
            while True:
                #main loop
                if userInput == 'exit':
                        print(f"<{self.name}>: Goodbye!")
                        break
                print(f"<{self.name}>: How can I help you today?")
                print(f"\033[3m\033[34mNot sure what to ask?\nTry:\n{random.choice(prompts)}\033[0m")
                while True:
                    #conversation loop
                    userInput = input(f"<{userName}>: ").lower()

                    if userInput == 'exit':
                        break

                    output = find_response(userInput)
                    while not output:
                        print(f"<{self.name}>: I didn't understand that, can you say it again?")
                        userInput = input(f"<{user[1]}>: ").lower()
                        output = find_response(userInput)
                        if output:
                            break
                    print(f"<{self.name}>: {output}")
        finally:
            database.connection.close()