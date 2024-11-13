import sqlite3

class Database:
    def __init__(self, chatbot):
        self.chatbot = chatbot+'.db'
        self.connection = sqlite3.connect(self.chatbot)
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute('''CREATE TABLE UserInfo (userID integer primary key, name text, nickname text, age integer, gender text)''')
            self.connection.commit()
        except:
            pass
        self.connection.close()
    
    def connect(self):
        self.connection = sqlite3.connect(self.chatbot)
        self.cursor = self.connection.cursor()

    def disconnect(self):
        self.connection.close()

    def sqlexecute(self, text):
        self.cursor.execute(text)
        self.connection.commit()