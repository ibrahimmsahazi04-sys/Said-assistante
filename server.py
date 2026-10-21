"""Assistant vocal : serveur Python gratuit (FastAPI + LLM compatible OpenAI + DuckDuckGo)."""
import os, re, json, subprocess
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from openai import OpenAI
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

# --- Configuration (variables d'environnement) ---
# Par défaut : Ollama en local (100 % gratuit). Pour Groq (gratuit) :
#   LLM_BASE_URL=https://api.groq.com/openai/v1  LLM_MODEL=openai/gpt-oss-120b  LLM_API_KEY=gsk_...
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
API_KEY = os.getenv("LLM_API_KEY", "ollama")
TOKEN = os.getenv("ASSISTANT_TOKEN", "changeme")
WORKDIR = os.path.abspath(os.getenv("WORKDIR", "./workspace"))
os.makedirs(WORKDIR, exist_ok=True)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
app = FastAPI()
sessions = {}

SYSTEM = (
    "Tu es Saïd, un assistant vocal en français. Tu appelles toujours l'utilisateur « monsieur ». Réponds de façon courte et naturelle, car ta réponse est lue à voix haute. "
    "Utilise web_search pour toute information récente. Pour écrire du code, donne-le dans un bloc ``` avec une phrase d'explication. "
    "Utilise run_command pour exécuter une commande shell : l'utilisateur la validera avant exécution. N'écris jamais d'astérisques ni de mise en forme markdown (pas de gras, pas de titres, pas de tableaux)."
)
SYSTEM_KM = (
    "Tu es Saïd, un assistant vocal. Tu réponds en shingazidja (le comorien de la Grande Comore), avec des phrases courtes et simples. "
    "Tu appelles l'utilisateur « bwana ». Tu connais mal cette langue : si tu hésites sur un mot, écris-le en shingazidja puis ajoute "
    "entre parenthèses sa traduction en français. Si l'utilisateur écrit en français, comprends-le mais réponds en shingazidja. "
    "Utilise web_search pour toute information récente, run_command pour exécuter une commande shell (l'utilisateur la validera), "
    "et donne le code dans un bloc ``` avec une phrase d'explication. N'écris jamais d'astérisques ni de mise en forme markdown (pas de gras, pas de titres, pas de tableaux)."
)
TOOLS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Cherche sur le web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "run_command", "description": "Exécute une commande shell dans le dossier de travail",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]

def clean(t):
    """Retire astérisques, titres et liens markdown (hors blocs de code)."""
    parts = t.split("```")
    for i in range(0, len(parts), 2):
        p = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", parts[i])
        p = re.sub(r"[*＊✱∗•`]+", "", p)
        parts[i] = re.sub(r"^#+\s*", "", p, flags=re.M)
    return "```".join(parts)

def web_search(query):
    try:
        res = DDGS().text(query, max_results=5)
        return "\n".join(f"- {r['title']}: {r['body']} ({r['href']})" for r in res) or "Aucun résultat."
    except Exception as e:
        return f"Erreur de recherche : {e}"

def run_command(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30, cwd=WORKDIR)
        return ((p.stdout + p.stderr) or "(aucune sortie)")[:3000]
    except subprocess.TimeoutExpired:
        return "Commande interrompue après 30 secondes."

def check(token):
    if token != TOKEN:
        raise HTTPException(401, "Code d'accès invalide")

def loop(sid):
    s = sessions[sid]
    try:
        for _ in range(6):
            m = client.chat.completions.create(model=MODEL, messages=s["messages"], tools=TOOLS).choices[0].message
            msg = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [{"id": t.id, "type": "function",
                    "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in m.tool_calls]
            s["messages"].append(msg)
            if not m.tool_calls:
                return {"type": "reply", "text": clean(m.content or "")}
            pending = None
            for t in m.tool_calls:
                args = json.loads(t.function.arguments or "{}")
                if t.function.name == "run_command":
                    if pending is None:
                        pending = {"id": t.id, "command": args.get("command", "")}
                    else:
                        s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": "Ignoré : une commande à la fois."})
                else:
                    s["messages"].append({"role": "tool", "tool_call_id": t.id, "content": web_search(args.get("query", ""))})
            if pending:
                s["pending"] = pending
                return {"type": "confirm", "command": pending["command"]}
        return {"type": "reply", "text": "Je n'ai pas réussi à terminer cette tâche."}
    except Exception as e:
        return {"type": "reply", "text": f"Erreur du modèle : {e}"}

class Chat(BaseModel):
    sid: str
    text: str
    lang: str = "fr"

class Confirm(BaseModel):
    sid: str
    approve: bool

@app.post("/api/chat")
def chat(b: Chat, x_token: str = Header("")):
    check(x_token)
    s = sessions.setdefault(b.sid, {"messages": [{"role": "system", "content": SYSTEM}], "pending": None})
    s["messages"][0]["content"] = SYSTEM_KM if b.lang == "km" else SYSTEM
    s["messages"].append({"role": "user", "content": b.text})
    return loop(b.sid)

@app.post("/api/confirm")
def confirm(b: Confirm, x_token: str = Header("")):
    check(x_token)
    s = sessions.get(b.sid)
    if not s or not s["pending"]:
        return {"type": "reply", "text": "Aucune commande en attente."}
    p, s["pending"] = s["pending"], None
    out = run_command(p["command"]) if b.approve else "Refusé par l'utilisateur."
    s["messages"].append({"role": "tool", "tool_call_id": p["id"], "content": out})
    r = loop(b.sid)
    r["output"] = out if b.approve else None
    return r

@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
