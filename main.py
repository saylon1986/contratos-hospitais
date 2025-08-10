import os
import json
import ast
import difflib
import pandas as pd
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from unidecode import unidecode

# ========= Config =========
DB_PATH = os.getenv("HOSPITAIS_CSV", "hospitais_BD_final.csv")

_df_cache: Optional[pd.DataFrame] = None

# ========= Utils =========
def load_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        try:
            _df_cache = pd.read_csv(DB_PATH, dtype=str)
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail=f"CSV não encontrado em '{DB_PATH}'.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Falha ao ler CSV: {e}")
    return _df_cache

def norm(s: Any) -> str:
    return unidecode(str(s or "")).strip().lower()

def safe_parse_json(texto: Any) -> Any:
    if texto is None:
        return None
    txt = str(texto).strip()
    if not txt:
        return None
    try:
        return json.loads(txt)
    except Exception:
        try:
            return ast.literal_eval(txt)
        except Exception:
            return {"_raw": txt, "_parsed": False}

def detectar_colunas_json(df: pd.DataFrame, limite: int = 200) -> List[str]:
    """Detecta colunas que parecem conter JSON em pelo menos uma linha."""
    cols: List[str] = []
    for c in df.columns:
        amostra = df[c].dropna().astype(str).head(limite)
        for v in amostra:
            v = v.strip()
            if v.startswith("{") or v.startswith("["):
                cols.append(c)
                break
    return cols

def selecionar_por_nome(
    df: pd.DataFrame,
    nome_informado: str,
    max_sugestoes: int = 5,
    auto_picking: bool = True,
    score_min: float = 0.85,
) -> Dict[str, Any]:
    """Fluxo: exato → contains → fuzzy (com auto-pick se score alto)."""
    if "nome_hospital" not in df.columns:
        return {"status": "nao_encontrado", "opcoes": [], "erro": "Coluna 'nome_hospital' não existe no CSV."}

    alvo = norm(nome_informado)
    nomes = df["nome_hospital"].fillna("")
    nomes_norm = nomes.map(norm)

    # 1) Exato (normalizado)
    sub = df[nomes_norm == alvo]
    if not sub.empty:
        return {"status": "selecionado", "row": sub.iloc[0].to_dict()}

    # 2) Contains (parcial)
    mask_contains = nomes_norm.str.contains(alvo, regex=False, na=False)
    sub_contains = df[mask_contains]
    if len(sub_contains) == 1:
        return {"status": "selecionado", "row": sub_contains.iloc[0].to_dict()}
    if len(sub_contains) > 1:
        return {"status": "desambiguar", "opcoes": sub_contains["nome_hospital"].head(max_sugestoes).tolist()}

    # 3) Fuzzy
    candidatos = nomes.tolist()
    proximos = difflib.get_close_matches(nome_informado, candidatos, n=max_sugestoes, cutoff=0.0)

    scored = []
    for cand in proximos:
        s = difflib.SequenceMatcher(a=norm(nome_informado), b=norm(cand)).ratio()
        scored.append((cand, s))
    scored.sort(key=lambda x: x[1], reverse=True)

    if scored:
        top_nome, top_score = scored[0]
        if auto_picking and top_score >= score_min:
            linha = df[df["nome_hospital"] == top_nome].iloc[0].to_dict()
            return {"status": "selecionado", "row": linha}
        return {"status": "desambiguar", "opcoes": [n for n, _ in scored]}

    return {"status": "nao_encontrado", "opcoes": []}

# ========= Pydantic =========
class Selector(BaseModel):
    nome_hospital: str

class AvaliarVariaveisRequest(BaseModel):
    selector: Selector
    # Se não vier, autodetecta as colunas JSON
    variaveis: Optional[List[str]] = None

# ========= FastAPI App =========
app = FastAPI(title="Ações do Banco de Hospitais", version="1.0.0")

# CORS liberado (útil p/ testes/tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/health", "/listar_hospitais", "/list_json_vars", "/avaliar_variaveis_json"]}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/list_json_vars")
def list_json_vars():
    df = load_df()
    return {"json_vars": detectar_colunas_json(df)}

@app.get("/listar_hospitais")
def listar_hospitais(q: Optional[str] = Query(default=None, description="Filtro por nome (parcial)"),
                     limit: int = Query(default=500, ge=1, le=10000)):
    df = load_df()
    if "nome_hospital" not in df.columns:
        return {"total": 0, "items": []}

    nomes = df["nome_hospital"].dropna().astype(str)

    if q and q.strip():
        alvo = norm(q)
        nomes_norm = nomes.map(norm)
        mask = nomes_norm.str.contains(alvo, regex=False, na=False)
        nomes = nomes[mask]

    nomes = nomes.drop_duplicates().sort_values().head(limit)
    return {"total": int(nomes.shape[0]), "items": nomes.tolist()}

@app.post("/avaliar_variaveis_json")
def avaliar_variaveis_json(body: AvaliarVariaveisRequest):
    """
    Body exemplo:
    {
      "selector": { "nome_hospital": "Sorocaba" },
      "variaveis": ["Metas", "Pagamento"]   # opcional; se faltar, autodetecta colunas JSON
    }
    """
    df = load_df()

    sel = selecionar_por_nome(
        df,
        body.selector.nome_hospital,
        max_sugestoes=5,
        auto_picking=True,
        score_min=0.85,
    )

    if sel.get("erro"):
        return {"ok": False, "error": sel["erro"]}

    if sel["status"] == "selecionado":
        row = sel["row"]
    elif sel["status"] == "desambiguar":
        return {"ok": False, "error": "Foram encontradas múltiplas opções. Escolha uma.", "sugestoes": sel["opcoes"]}
    else:
        return {"ok": False, "error": "Hospital não encontrado.", "sugestoes": sel["opcoes"]}

    # Variáveis a ler: se não vieram, autodetecta
    variaveis = body.variaveis if body.variaveis else detectar_colunas_json(df)

    resultados: Dict[str, Any] = {}
    for var in variaveis:
        if var not in row:
            resultados[var] = {"_error": "Variável inexistente nesta base."}
            continue
        val = row[var]
        if val is None or str(val).strip() == "":
            resultados[var] = {"_error": "Variável vazia."}
            continue
        resultados[var] = safe_parse_json(val)

    return {"ok": True, "hospital": {"nome_hospital": row.get("nome_hospital")}, "resultados": resultados}

