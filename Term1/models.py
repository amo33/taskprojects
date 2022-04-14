from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
# Use this file to define model 
class User(db.Model):
    
    idx = db.Column(db.Integer, primary_key=True) # db.Column makes restraint on each columns
    username = db.Column(db.String(64), index=True)
    age = db.Column(db.Integer, primary_key = False)
    Image = db.Column(db.String(512))
    
    def __repr__(self):
        #lines = []
        #lines.append(self.username)
        #lines.append(self.age)
        #lines.append(self.Image)
        #return lines
        line = {}
        line['ID'] = self.idx
        line['USER'] = self.username
        line['Age'] = self.age
        line['Image'] = self.Image
        return str(line)

