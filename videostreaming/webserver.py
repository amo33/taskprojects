from flask import Flask, render_template
from flask_socketio import SocketIO
import io
app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*") #flask-socketio polling 에러  
#https://github.com/miguelgrinberg/python-socketio/issues/403

@app.route('/') #html 보여준다.
def index():
    return render_template('index.html')

@socketio.on('streaming') # streaming 이라는 이벤트가 client로부터 emit 되면 이 함수를 실행 
def connect(framedata):

    socketio.emit('sending',framedata)


if __name__ == '__main__':
    socketio.run(app, debug=True)