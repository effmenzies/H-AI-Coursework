import nltk
import bs4 as bs
from bs4 import BeautifulSoup as bsoup
from urllib import request
import sqlite3
import json
import time
from nltk import *
from database import *
from similarity import *
from qanda import *
from smalltalk import *

class Chatbot:
    def __init__(self, database):
        self.name = database
        nltk.download('wordnet')
        build_dt_matrix(questions)
        create_smalltalk()

    
    def main(self):
        database = Database(self.name)
        database.connect()
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
                userInput = input(f"<{user[1]}>: ")
                if userInput.lower() == 'exit':
                    print(f"<{self.name}>: Goodbye!")
                    break
                if userInput.lower() == "question":
                    print(f"<{self.name}>: What would you like to know?")
                    userInput = input(f"<{user[1]}>: ")
                    answer = qanda_search(userInput)
                    time.sleep(1)
                    print(f"<{self.name}>: {answer}")
                
        finally:
            database.connection.close()