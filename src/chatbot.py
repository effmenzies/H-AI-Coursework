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
from intent import *

class Chatbot:
    def __init__(self, database):
        self.name = database
        self.sentiment = 0
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger_eng')
        intent_init()
        qanda_init()
        clf_init()
    
    def main(self):
        database = Database(self.name)
        database.connect()
        prompts = ["How are you today?","What's the weather like today?","What's the time?","What can you do for me?",
                   "What's your favourite food?","Recommend things to do near me."]
        tasks = ["Have a chat","Answer questions","Recommend music","Make a booking","Set a reminder"]
        try:
            userName=None
            userInput=None
            print(f"<{self.name}>: Hi! My name is {self.name}.")

            #check if user  1 exists
            database.sqlexecute('''SELECT * FROM UserInfo WHERE userID=1''')
            user = database.cursor.fetchone()
            if user:
                #greet user
                userName = user[2]
                print(f"\nTo exit the chat, type 'exit'\n")
                print(f"<{self.name}>: Hi {userName}!")
            else:
                while not userName:
                    #ask for name
                    print(f"What's your name?")
                    userInput = input("<you>: ")
                    names = extract_names(userInput)
                    fullName = max(names).capitalize()
                    nickName = None
                    print(f"<{self.name}>: Hi, I've got your full name as {fullName}.")
                    if len(names)>1:
                        nickName = min(names).capitalize()
                        print(f"And {nickName} as a nickname.")
                    print(f"Is this correct?")
                    userInput = input("<you>: ")
                    if cosine_sim(userInput, 'confirmations') == 'yes':
                        while not nickName:
                            print(f"<{self.name}>: Is {fullName} your preferred name?")
                            userInput = input("<you>: ")
                            if cosine_sim(userInput, 'confirmations') == 'yes':
                                nickName=fullName
                            elif cosine_sim(userInput, 'confirmations') == 'no':
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
                #introductory loop
                if userInput == 'exit':
                        print(f"<{self.name}>: Goodbye!")
                        break
                print(f"<{self.name}>: How can I help you today?")
                print(f"\033[3m\033[34mNot sure what to ask?\nTry:\n{random.choice(prompts)}\033[0m")
                userInput = input(f"<{userName}>: ").lower()
                while True:
                    #main loop
                    if userInput == 'exit':
                        break
                    intent = classify(userInput)
                    while intent=='smalltalk':
                        #smalltalk loop
                        i = cosine_sim(userInput,'greetings') #check intent
                        if i == 'discover': #select an option from a list
                            print(f"<{self.name}>: These are some of the things I can do for you:\033[3m\033[34m")
                            for t in tasks:
                                print(f"\n{t}")
                            print(f"\033[0m\nWhat would you like to do?")
                            while True:
                                #choose an option loop
                                userInput = input(f"<{userName}>: ").lower()
                                if userInput == 'exit':
                                    break
                                elif userInput in 'have a chat':
                                    intent = 'smalltalk'
                                    print(f"<{self.name}>: Certainly.\nHow is your day going?")
                                    userInput = input(f"<{userName}>: ").lower()
                                    break
                                elif userInput in 'answer questions':
                                    intent = 'qanda'
                                    print(f"<{self.name}>: What would you like to know?")
                                    userInput = input(f"<{userName}>: ").lower()
                                    break
                                else:
                                    print(f"<{self.name}>: I didn't understand that, can you say it again?") #loops until an option is selected or user exits
                        elif i == 'me':
                            #get user info from database
                            if 'name' in userInput:
                                name, nickname = database.get_name()
                                if name==nickname:
                                    print(f"<{self.name}>: Your name is {name}. You don't have a nickname, would you like one?")
                                    userInput = input(f"<{userName}>: ").lower()
                                    if cosine_sim(userInput, 'confirmations') == 'no':
                                        print(f"<{self.name}>: No worries!")
                                    else:
                                        while True:
                                            #set nickname loop
                                            print(f"<{self.name}>: What do you want to be called?")
                                            userInput = input(f"<{userName}>: ")
                                            name = extract_names(userInput)[0].capitalize()
                                            print(f"<{self.name}>: Ok, do you want me to call you {name} from now on?")
                                            userInput = input(f"<{userName}>: ").lower()
                                            if cosine_sim(userInput, 'confirmations') == 'yes':
                                                print(f"<{self.name}>: No worries!")
                                                database.sqlexecute(f'''UPDATE userInfo SET nickname = ('{name}') WHERE userID=1''')
                                                userName = name
                                                userInput = input(f"<{userName}>: ")
                                                break
                                            else: print(f'<{self.name}>:Sorry I misunderstood.')
                                        break
                                else:
                                    print(f"<{self.name}>: Your name is {nickname}, but your full name is {name}.")
                                    userInput = input(f"<{userName}>: ")
                                    break
                            elif bool({'old','age','birthday'}&set(extract_info(userInput))):
                                age, birthday = database.get_age()
                                if not age and not birthday:
                                    print(f"<{self.name}>: I don't know! When is your birthday?")
                                    userInput = input(f"<{userName}>: ").lower()
                        intent=classify(userInput)
                        if intent=='smalltalk': #incase intent changed
                            output = find_response(userInput)
                            print(f"<{self.name}>: {output}")
                            userInput = input(f"<{userName}>: ").lower()
                            intent = classify(userInput) #update condition
                        if userInput == 'exit':
                            break
                        
                    while intent=='qanda':
                        #qanda loop
                        intent=classify(userInput)
                        if intent=='qanda':
                            output = cosine_sim_answer(userInput,'qanda')
                            print(f"<{self.name}>: {output}")
                            userInput = input(f"<{userName}>: ").lower()
                            if userInput == 'exit':
                                break

        finally:
            database.connection.close()

'''
                        elif i == 'you':
                            print(f"<{self.name}>: So you want to talk about me.")
                            #talk about the bot
                        '''