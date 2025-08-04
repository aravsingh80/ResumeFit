from app.routes import app  # moved after env var is set

if __name__ == "__main__":
    app.run(debug=True)
