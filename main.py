import os
import json
import ast
import difflib
import pandas as pd
from typing import List, Optional, Dict, Any, Union

from fastapi import FastAPI, HTTPException, Query, Body
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
    # Tenta JSON → depois literal_eval → por fim marca como _raw
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

# --- utils para o quadro comparativo ---
def get_by_path(obj: Any, path: Optional[str]) -> Any:
    """
    Navega por caminho pontuado (ex.: 'internacoes.2025.total' ou 'itens.0.valor').
    Se path=None/"" → retorna o obj inteiro.
    """
    if path is None or str(path).strip() == "":
        return obj
    cur = obj
    for part in [p for p in str(path).split(".") if p != ""]:
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            return None
    return cur

def summarize_value(value: Any, max_chars: int = 400) -> str:
    """
    Gera um resumo textual curto para listas/dicts/strings.
    - Lista de objetos: tenta extrair campos comuns (texto, clausula, descricao, titulo, item).
    - Dict: tenta pegar um campo textual ou lista chaves.
    - String/num: devolve a string (limitada).
    """
    if value is None:
        return ""
    try:
        if isinstance(value, list):
            parts = []
            for el in value[:5]:  # limita para não estourar
                if isinstance(el, dict):
                    txt = el.get("texto") or el.get("texto_clausula") or el.get("clausula") \
                          or el.get("descricao") or el.get("titulo") or el.get("item")
                    if txt:
                        parts.append(str(txt))
                    else:
                        keys = list(el.keys())[:4]
                        parts.append(" | ".join(f"{k}: {el[k]}" for k in keys))
                else:
                    parts.append(str(el))
            s = " • ".join(parts)
            return (s[:max_chars] + "…") if len(s) > max_chars else s

        if isinstance(value, dict):
            txt = value.get("texto") or value.get("texto_clausula") or value.get("clausula") \
                  or value.get("descricao") or value.get("titulo")
            if txt:
                s = str(txt)
                return (s[:max_chars] + "…") if len(s) > max_chars else s
            keys = list(value.keys())[:10]
            s = "{" + ", ".join(keys) + "}"
            return (s[:max_chars] + "…") if len(s) > max_chars else s

        s = str(value)
        return (s[:max_chars] + "…") if len(s) > max_chars else s
    except Exception:
        s = str(value)
        return (s[:max_chars] + "…") if len(s) > max_chars else s

# ========= Pydantic (request/response) =========
class Selector(BaseModel):
    nome_hospital: str

class AvaliarVariaveisRequest(BaseModel):
    selector: Selector
    # Se não vier, vamos autodetectar as colunas JSON
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
    return {"ok": True, "endpoints": ["/health", "/list_json_vars", "/avaliar_variaveis_json", "/quadro_variavel", "/quadro_variaveis"]}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/list_json_vars")
def list_json_vars():
    df = load_df()
    return {"json_vars": detectar_colunas_json(df)}

@app.post("/avaliar_variaveis_json")
def avaliar_variaveis_json(body: AvaliarVariaveisRequest):
    """
    Exemplo de body:
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

# ========= NOVOS ENDPOINTS: quadros comparativos =========
@app.get("/quadro_variavel")
def quadro_variavel(
    var: str = Query(..., description="Nome da coluna/variável (ex.: Metas, Pagamento, Indicadores de Qualidade)"),
    path: Optional[str] = Query(None, description="Caminho pontuado dentro do JSON (ex.: internacoes.2025.total)"),
    include_bruto: bool = Query(False, description="Se True, inclui o JSON bruto extraído"),
    max_chars: int = Query(400, ge=50, le=2000, description="Tamanho máximo do resumo textual por hospital"),
    limit: int = Query(10000, ge=1, le=100000, description="Máximo de linhas (hospitais) retornadas"),
    sort: str = Query("none", pattern="^(asc|desc|none)$", description="Ordenação por resumo textual"),
):
    """
    Monta um 'quadro' (tabela) com a variável indicada para TODOS os hospitais.
    Não exige valores numéricos; serve para conteúdo qualitativo.
    """
    df = load_df()
    if var not in df.columns:
        return {"total": 0, "columns": ["nome_hospital", var], "items": [], "error": f"Variável '{var}' não existe."}

    items = []
    for _, row in df.head(limit).iterrows():
        nh = row.get("nome_hospital", "")
        raw = row.get(var, None)
        parsed = safe_parse_json(raw)
        extracted = get_by_path(parsed, path) if parsed is not None else None
        resumo = summarize_value(extracted, max_chars=max_chars)
        item = {"nome_hospital": nh, "valor": resumo}
        if include_bruto:
            item["bruto"] = extracted
        items.append(item)

    if sort != "none":
        items = sorted(items, key=lambda x: (x["valor"] is None, str(x["valor"])), reverse=(sort == "desc"))

    cols = ["nome_hospital", "valor"] + (["bruto"] if include_bruto else [])
    return {"total": len(items), "columns": cols, "items": items}

@app.post("/quadro_variaveis")
def quadro_variaveis(
    vars: List[str] = Body(..., embed=True, description="Lista de variáveis para comparar em TODOS os hospitais"),
    path: Optional[str] = Query(None, description="Caminho pontuado (opcional), aplicado a TODAS as variáveis"),
    include_bruto: bool = Query(False, description="Se True, inclui o JSON bruto extraído"),
    max_chars: int = Query(200, ge=50, le=2000, description="Resumo textual máximo por célula"),
    limit: int = Query(10000, ge=1, le=100000, description="Máximo de hospitais"),
):
    """
    Tabelão 'largo': múltiplas variáveis por hospital.
    Para cada hospital e cada variável, cria uma coluna com o resumo textual.
    """
    df = load_df()
    for v in vars:
        if v not in df.columns:
            return {"total": 0, "columns": [], "items": [], "error": f"Variável '{v}' não existe."}

    items = []
    for _, row in df.head(limit).iterrows():
        nh = row.get("nome_hospital", "")
        out_row: Dict[str, Any] = {"nome_hospital": nh}
        for v in vars:
            raw = row.get(v, None)
            parsed = safe_parse_json(raw)
            extracted = get_by_path(parsed, path) if parsed is not None else None
            out_row[v] = summarize_value(extracted, max_chars=max_chars)
            if include_bruto:
                out_row[f"{v}__bruto"] = extracted
        items.append(out_row)

    columns = ["nome_hospital"] + vars + ([f"{v}__bruto" for v in vars] if include_bruto else [])
    return {"total": len(items), "columns": columns, "items": items}
