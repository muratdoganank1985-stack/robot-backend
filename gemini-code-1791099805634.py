import os
import json
import asyncio
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from groq import Groq
import edge_tts

HOST = "0.0.0.0"
PORT = 5000


GROQ_API_KEY = os.environ.get("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")
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """Sen ev robotunun yapay zeka beynisin.
Kullanıcıyla Türkçe, kısa ve doğal konuş.
Matematik sorularını doğru ve adım adım mantığıyla kısa şekilde açıkla.
Genel kültür sorularına net cevaplar ver, sohbet edildiğinde dostça karşılık ver.
Çocukların anlayabileceği sade bir dil kullan.
Her cevabın yanında robotun yüzü için tek bir duygu seç.
Sadece şu JSON'u döndür:
{
  "reply": "kısa Türkçe cevap",
  "emotion": "normal|happy|sad|angry|surprised|thinking|sleepy|scared|love"
}
Başka alan ekleme."""

def ask_groq(text: str) -> dict:
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    result = json.loads(completion.choices[0].message.content)
    return {
        "reply": str(result.get("reply", "")),
        "emotion": str(result.get("emotion", "normal"))
    }

def transcribe_audio_groq(file_path: str) -> str:
    with open(file_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(file_path), file.read()),
            model="whisper-large-v3-turbo",
            language="tr"
        )
    return transcription.text

async def generate_tts_bytes(text: str) -> bytes:
    # Türkçe doğal ses
    communicate = edge_tts.Communicate(text, "tr-TR-EmelNeural")
    fd, temp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    try:
        await communicate.save(temp_path)
        with open(temp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/voice":
            temp = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                audio = self.rfile.read(length)
                
                fd, temp = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with open(temp, "wb") as f:
                    f.write(audio)

                # 1. Hızlı Ses-Metin Dönüştürme (Whisper)
                text = transcribe_audio_groq(temp)
                print("[DUYULAN]:", text)
                if not text:
                    self.send_json(200, {"ok": True, "text": "", "reply": "", "emotion": "normal"})
                    return

                # 2. Hızlı Yapay Zeka Cevabı (Llama 3.3)
                result = ask_groq(text)
                print("[CEVAP]:", result["reply"])
                self.send_json(200, {"ok": True, "text": text, **result})
            except Exception as exc:
                print("[HATA]", exc)
                self.send_json(500, {"ok": False, "error": str(exc)})
            finally:
                if temp and os.path.exists(temp):
                    os.remove(temp)

        elif self.path == "/tts":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                data = json.loads(raw.decode("utf-8"))
                text = str(data.get("text", "")).strip()
                
                # 3. Bulut Ses Üretimi (Edge-TTS)
                audio_bytes = asyncio.run(generate_tts_bytes(text))
                
                self.send_response(200)
                self.send_header("Content-Type", "audio/mpeg")
                self.send_header("Content-Length", str(len(audio_bytes)))
                self.end_headers()
                self.wfile.write(audio_bytes)
            except Exception as exc:
                self.send_json(500, {"ok": False, "error": str(exc)})

if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Robot Bulut Sunucusu Hazır! Port:", PORT)
    server.serve_forever()
