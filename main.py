import pandas as pd
import json, ast
from fastapi import FastAPI, Request
from unidecode import unidecode

DB_PATH = "hospitais_BD_final.csv"  # o CSV vai junto no repo
df_cache = None

def load_df():
    global df_cache
    if df_cache is None:
        df_cache = pd.read_csv(DB_PATH, dtype=str)
    return df_cache

def norm(s):
    return unidecode(str(s or "")).strip().lower()

app = FastAPI(title="Ações do Banco de Hospitais", version="1.0.0")

@app.post("/avaliar_variaveis_json")
async def avaliar_variaveis_json(request: Request):
    body = await request.json()
    selector = body.get("selector", {})
    variaveis = body.get("variaveis", [])
    df = load_df()

    if "cnes" in selector and "cnes" in df.columns:
        sub = df[df["cnes"].astype(str) == str(selector["cnes"])]
    elif "nome_hospital" in selector and "nome_hospital" in df.columns:
        sub = df[norm(df["nome_hospital"]) == norm(selector["nome_hospital"])]
    else:
        return {"ok": False, "error": "Informe 'cnes' (se existir) ou 'nome_hospital'."}

    if sub.empty:
        return {"ok": False, "error": "Hospital não encontrado."}

    row = sub.iloc[0].to_dict()
    resultados = {}
    for var in variaveis or []:
        if var not in row:
            resultados[var] = {"_error": "Variável inexistente nesta base."}
            continue
        val = row[var]
        if val is None or str(val).strip() == "":
            resultados[var] = {"_error": "Variável vazia."}
            continue
        txt = str(val).strip()
        try:
            resultados[var] = json.loads(txt)
        except Exception:
            try:
                resultados[var] = ast.literal_eval(txt)
            except Exception:
                resultados[var] = {"_raw": txt, "_parsed": False}

    return {
        "ok": True,
        "hospital": {"nome_hospital": row.get("nome_hospital"), "cnes": row.get("cnes")},
        "resultados": resultados
    }
