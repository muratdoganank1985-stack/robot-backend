import os
import json
import asyncio
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from groq import Groq
import edge_tts

# Render'ın dinamik PORT ayarı
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))

# Groq API Key ayarı
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
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

async def generate_speech_bytes(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "tr-TR-AhmetNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp_path = tmp.name
    await communicate.save(tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return data

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Robot Backend is Live!".encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        
        try:
            req_data = json.loads(body)
            user_text = req_data.get("text", "")
            
            # 1. Groq'tan yanıt ve duygu al
            ai_res = ask_groq(user_text)
            reply_text = ai_res["reply"]
            emotion = ai_res["emotion"]
            
            # 2. Edge-TTS ile ses üret
            audio_bytes = asyncio.run(generate_speech_bytes(reply_text))
            
            # 3. Yanıtı gönder
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Robot-Emotion", emotion)
            self.send_header("X-Robot-Reply", reply_text.encode("utf-8").decode("latin-1", "ignore"))
            self.end_headers()
            self.wfile.write(audio_bytes)

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_msg = json.dumps({"error": str(e)})
            self.wfile.write(err_msg.encode("utf-8"))

if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"Server started at http://{HOST}:{PORT}")
    server.serve_forever()
