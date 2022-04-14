"""
import sqlite3
import sqlite_connector
from datetime import datetime

class DatabaseManager(sqlite_connector.DatabaseManager):
    ''' Database Manager '''

    def __init__(self, db_name):
        super().__init__(db_name)

    def insert_record_and_get_insertedid(self):
        '''Insert a row inside table named 'article' '''

        userid = 1107
        name = 'Python Insert Multiple Records'
        category = 'Python'
        views = likes = dislikes = comments = 0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = "INSERT INTO article values (?, ?, ?, ?, ?, ?, ?, ?)"

        self.cur.execute(query, (userid, name, category, views, likes, dislikes, comments, timestamp))
        print(f'Last row id : {self.cur.lastrowid}')
        self.conn.commit()

        print('Record inserted succesfully')

def main():
    ''' Insert a record in article table '''

    dbmngr = DatabaseManager('cppsecrets.db')
    dbmngr.create_connection()
    dbmngr.get_cursor()
    dbmngr.insert_record_and_get_insertedid()  # make sure table exists before doing this
    dbmngr.close_connection()

if __name__ == '__main__':
    main()
"""