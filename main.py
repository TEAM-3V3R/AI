from flask import Flask
from prompt import prompt_bp
from sam import sam_bp
from fastSAM import fastsam_bp
from prompt_analyzer.analyzer import analyzer_bp
from prompt_analyzer.categories import categories_bp

app = Flask(__name__)

# 각 기능 모듈의 Blueprint 등록
app.register_blueprint(prompt_bp)
app.register_blueprint(sam_bp)
app.register_blueprint(fastsam_bp)
app.register_blueprint(analyzer_bp)
app.register_blueprint(categories_bp)

@app.route("/", methods=["GET"])
def index():
    return "API 서버가 정상적으로 실행 중입니다"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
