from flask import Flask, request, jsonify
import os
import uuid
import whisper
import time
from jiwer import wer

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("base")

@app.route("/avaliar", methods=["POST"])
def avaliar():
    audio = request.files['audio']
    texto_original = request.form['texto']
    filename = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.wav")
    audio.save(filename)

    result = model.transcribe(filename)
    transcricao = result['text']
    tempo_total = result['segments'][-1]['end'] if result['segments'] else 0
    tempo_total = max(tempo_total, 1)

    precisao = (1 - wer(texto_original.lower(), transcricao.lower())) * 100
    velocidade = len(transcricao.split()) / (tempo_total / 60)

    pausas = [seg["end"] - seg["start"] for seg in result.get("segments", [])]
    prosodia = "Boa" if all(0.3 <= p <= 3.0 for p in pausas) else "Irregular"

    return jsonify({
        "transcricao": transcricao.strip(),
        "precisao": round(precisao, 2),
        "velocidade": round(velocidade, 2),
        "prosodia": prosodia
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
