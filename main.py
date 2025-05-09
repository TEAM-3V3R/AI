from flask import Flask
from prompt import prompt_bp
from morpheme import morpheme_bp
from homonym import homonym_bp

app = Flask(__name__)

# 각 기능 모듈의 Blueprint 등록
app.register_blueprint(prompt_bp)
app.register_blueprint(morpheme_bp)
app.register_blueprint(homonym_bp)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
