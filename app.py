import os
import json
import asyncio
import tempfile
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import edge_tts
from pydub import AudioSegment

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 5000))
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def process_voice_input(wav_bytes: bytes) -> dict:
    if not GROQ_API_KEY:
        print("[HATA]: GROQ_API_KEY bulunamadi!")
        return {
            "text": "API Key Yok",
            "reply": "Render panelinde GROQ API anahtari eksik kanka!",
            "emotion": "angry"
        }

    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(tmp_path), file.read()),
                    model="whisper-large-v3-turbo",
                    language="tr"
                )
            user_text = transcription.text
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print(f"[STT ALINDI]: {user_text}")

        if not user_text or not user_text.strip():
            return {
                "text": "(Ses Algilanmadi)",
                "reply": "Seni duyamadim kanka, tekrar soyler misin?",
                "emotion": "surprised"
            }

        SYSTEM_PROMPT = """Sen sevimli bir ev robotusun.
Kullaniciyla Turkce, kisa ve samimi konus.
Cevabina uygun tek bir duygu sec.
Sadece su JSON formatinda cevap ver:
{"reply": "kisa cevap", "emotion": "normal|happy|sad|angry|surprised|thinking|sleepy|scared|love"}"""

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"}
        )
        
        raw_res = completion.choices[0].message.content
        result = json.loads(raw_res)
        return {
            "text": user_text,
            "reply": str(result.get("reply", "Anlamadim kanka.")),
            "emotion": str(result.get("emotion", "normal"))
        }

    except Exception as e:
        print(f"[PROCESS ERROR]: {e}")
        return {
            "text": "Sunucu Hatasi",
            "reply": "Hata olustu kanka.",
            "emotion": "sad"
        }

async def generate_wav_bytes(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, "tr-TR-AhmetNeural")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_mp3:
        mp3_path = tmp_mp3.name

    await communicate.save(mp3_path)

    try:
        audio = AudioSegment.from_file(mp3_path, format="mp3")
        audio = audio.set_frame_rate(22050).set_channels(1).set_sample_width(2)

        wav_io = BytesIO()
        audio.export(wav_io, format="wav")
        return wav_io.getvalue()
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)

class RequestHandler(BaseHTTPRequestHandler):
    # HTTP/1.1 chunked aktarım belasını engellemek için HTTP/1.0 zorluyoruz
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("Robot Backend is Live!".encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length)

        if self.path == "/voice":
            res_data = process_voice_input(body_bytes)
            res_json = json.dumps(res_data, ensure_ascii=False)
            response_bytes = res_json.encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        elif self.path == "/tts":
            try:
                req_data = json.loads(body_bytes.decode("utf-8"))
                reply_text = req_data.get("text", "Merhaba")
                wav_bytes = asyncio.run(generate_wav_bytes(reply_text))

                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(wav_bytes)))
                self.end_headers()
                self.wfile.write(wav_bytes)
            except Exception as e:
                print(f"[TTS ERROR]: {e}")
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), RequestHandler)
    print(f"Sunucu aktif: port {PORT}")
    server.serve_forever()
