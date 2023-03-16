from views import app
import os
app.secret_key = os.urandom(24)

if __name__=="__main__":
    app.run(debug=True, port=5050, host="0.0.0.0")