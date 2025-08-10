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


import re

NUM_RE = re.compile(r"(?<!\d)(\d{1,3}(?:\.\d{3})*|\d+)(?:,\d+)?")

def to_int_count(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    # remove decimais, mantemos contagem inteira
    s = s.replace(".", "")
    s = s.split(",")[0]
    try:
        return int(s)
    except Exception:
        return None

def flatten_text_items(value: Any) -> List[Dict[str, Any]]:
    """
    Converte o JSON (já extraído/unwrap) em lista de itens textuais com metadados.
    Cada item: {"texto": "...", "pagina": ..., "clausula": ...}
    """
    out: List[Dict[str, Any]] = []
    def add(txt, meta=None):
        meta = meta or {}
        out.append({
            "texto": str(txt) if txt is not None else "",
            "pagina": meta.get("pagina") or meta.get("page") or meta.get("pagina_encontrada"),
            "clausula": meta.get("clausula") or meta.get("numero") or meta.get("n_clausula"),
        })

    if isinstance(value, str):
        add(value, {})
    elif isinstance(value, dict):
        # se for um dict com campo textual
        txt = value.get("texto") or value.get("texto_clausula") or value.get("clausula") or value.get("descricao") or value.get("titulo")
        if txt:
            add(txt, value)
        # também varre listas internas comuns
        for k, v in list(value.items()):
            if isinstance(v, list):
                for el in v:
                    if isinstance(el, dict):
                        t = el.get("texto") or el.get("texto_clausula") or el.get("clausula") or el.get("descricao") or el.get("titulo")
                        if t:
                            add(t, el)
                    elif isinstance(el, str):
                        add(el, {})
    elif isinstance(value, list):
        for el in value:
            if isinstance(el, dict):
                t = el.get("texto") or el.get("texto_clausula") or el.get("clausula") or el.get("descricao") or el.get("titulo")
                if t:
                    add(t, el)
            elif isinstance(el, str):
                add(el, {})
    return out

def find_numbers_near_keywords(text: str, keywords: List[str], window: int = 60) -> List[Dict[str, Any]]:
    """
    Procura números próximos das keywords (antes/depois, janela em caracteres).
    Retorna lista de matches com {"num": int, "span": (i,j), "kw": <kw>, "contexto": <trecho>}
    """
    results: List[Dict[str, Any]] = []
    low = text.lower()
    for kw in keywords:
        q = kw.strip().lower()
        start = 0
        while True:
            idx = low.find(q, start)
            if idx == -1:
                break
            a = max(0, idx - window)
            b = min(len(text), idx + len(q) + window)
            trecho = text[a:b]
            # busca números no trecho
            for m in NUM_RE.finditer(trecho):
                n = to_int_count(m.group(1))
                if n is not None:
                    results.append({
                        "num": n,
                        "span": (a + m.start(), a + m.end()),
                        "kw": kw,
                        "contexto": trecho.strip()
                    })
            start = idx + len(q)
    return results



# ==== Heurísticas genéricas: auto-unwrap + escolha automática de path ====
def auto_unwrap(value, max_depth=2):
    """
    Se for dict com UMA chave, desce automaticamente (ex.: {'clausulas': [...]})
    até max_depth vezes.
    """
    cur = value
    depth = 0
    while isinstance(cur, dict) and len(cur.keys()) == 1 and depth < max_depth:
        (only_key,) = tuple(cur.keys())
        cur = cur[only_key]
        depth += 1
    return cur

TEXT_KEYS = {"texto","texto_clausula","clausula","descricao","descrição","titulo","título","item","resumo","conteudo"}

def is_texty(x):
    if x is None:
        return False
    if isinstance(x, str) and x.strip():
        return True
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, dict):
        return any(k in x for k in TEXT_KEYS)
    if isinstance(x, list):
        for el in x[:3]:
            if isinstance(el, str) and el.strip():
                return True
            if isinstance(el, dict) and any(k in el for k in TEXT_KEYS):
                return True
        return False
    return False

def iter_paths(obj, max_depth=3):
    """
    Gera caminhos candidatos de até max_depth (prioriza chaves comuns).
    """
    from collections import deque
    PRIORITY = ("clausulas","itens","metas","indicadores","lista","dados","conteudo","sections")
    q = deque([("", obj, 0)])
    while q:
        path, cur, d = q.popleft()
        yield path, cur
        if d >= max_depth:
            continue
        if isinstance(cur, dict):
            keys = list(cur.keys())
            keys.sort(key=lambda k: (k not in PRIORITY, k))
            for k in keys[:20]:
                q.append((f"{path}.{k}" if path else k, cur[k], d+1))
        elif isinstance(cur, list):
            for i, el in enumerate(cur[:5]):
                q.append((f"{path}.{i}" if path else str(i), el, d+1))

def pick_best_path(obj, max_depth=3):
    """
    Escolhe o melhor caminho 'textual' dentro de obj.
    Preferências: lista de dicts textuais > dict com campos textuais > lista de strings/números > simples.
    Retorna (best_path, best_value).
    """
    best = (None, None, -1)  # (path, value, score)
    for p, v in iter_paths(obj, max_depth=max_depth):
        score = 0
        if isinstance(v, list):
            if v and all(isinstance(el, dict) for el in v[:min(3, len(v))]):
                if any(any(k in el for k in TEXT_KEYS) for el in v[:min(5, len(v))]):
                    score = 3
                else:
                    score = 2
            elif any(isinstance(el, (str, int, float)) for el in v[:min(5, len(v))]):
                score = 2
        elif isinstance(v, dict):
            if any(k in v for k in TEXT_KEYS):
                score = 2
            else:
                score = 1
        elif isinstance(v, (str, int, float)) and str(v).strip():
            score = 1

        if score > best[2]:
            best = (p, v, score)
    return best[0], best[1]

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

# ========= NOVOS ENDPOINTS: quadros comparativos (com paginação offset/limit) =========
@app.get("/quadro_variavel")
def quadro_variavel(
    var: str = Query(..., description="Nome da coluna/variável (ex.: Metas, Pagamento, Indicadores de Qualidade)"),
    path: Optional[str] = Query(None, description="Caminho pontuado dentro do JSON (ex.: internacoes.2025.total)"),
    include_bruto: bool = Query(False, description="Se True, inclui o JSON bruto extraído"),
    max_chars: int = Query(400, ge=50, le=2000, description="Tamanho máximo do resumo textual por hospital"),
    limit: int = Query(200, ge=1, le=10000, description="Quantos hospitais retornar"),
    offset: int = Query(0, ge=0, description="Deslocamento (página) de hospitais"),
    sort: str = Query("none", pattern="^(asc|desc|none)$", description="Ordenação por resumo textual"),
):
    """
    Monta um 'quadro' (tabela) com a variável indicada para TODOS os hospitais.
    Não exige valores numéricos; serve para conteúdo qualitativo.
    Suporta paginação via offset/limit.
    """
    df = load_df()
    total_available = len(df)
    if var not in df.columns:
        return {"total": 0, "columns": ["nome_hospital", var], "items": [], "error": f"Variável '{var}' não existe.",
                "page": {"offset": offset, "limit": limit, "total_available": total_available}}

    # paginação
    df_page = df.iloc[offset: offset + limit]

    items = []
    for _, row in df_page.iterrows():
        nh = row.get("nome_hospital", "")
        raw = row.get(var, None)
        parsed = safe_parse_json(raw)
        if parsed is not None:
            if path and str(path).strip():
                extracted = get_by_path(parsed, path)
            else:
                unwrapped = auto_unwrap(parsed)
                bp, bv = pick_best_path(unwrapped, max_depth=3)
                extracted = bv if bp is not None else unwrapped
        else:
            extracted = None
        resumo = summarize_value(extracted, max_chars=max_chars)
        item = {"nome_hospital": nh, "valor": resumo}
        if include_bruto:
            item["bruto"] = extracted
        items.append(item)

    if sort != "none":
        items = sorted(items, key=lambda x: (x["valor"] is None, str(x["valor"])), reverse=(sort == "desc"))

    cols = ["nome_hospital", "valor"] + (["bruto"] if include_bruto else [])
    return {
        "total": len(items),
        "columns": cols,
        "items": items,
        "page": {"offset": offset, "limit": limit, "total_available": total_available}
    }

@app.post("/quadro_variaveis")
def quadro_variaveis(
    vars: List[str] = Body(..., embed=True, description="Lista de variáveis para comparar em TODOS os hospitais"),
    path: Optional[str] = Query(None, description="Caminho pontuado (opcional), aplicado a TODAS as variáveis"),
    include_bruto: bool = Query(False, description="Se True, inclui o JSON bruto extraído"),
    max_chars: int = Query(200, ge=50, le=2000, description="Resumo textual máximo por célula"),
    limit: int = Query(200, ge=1, le=10000, description="Quantos hospitais retornar"),
    offset: int = Query(0, ge=0, description="Deslocamento (página) de hospitais"),
):
    """
    Tabelão 'largo': múltiplas variáveis por hospital.
    Para cada hospital e cada variável, cria uma coluna com o resumo textual.
    Suporta paginação via offset/limit.
    """
    df = load_df()
    total_available = len(df)
    for v in vars:
        if v not in df.columns:
            return {"total": 0, "columns": [], "items": [], "error": f"Variável '{v}' não existe.",
                    "page": {"offset": offset, "limit": limit, "total_available": total_available}}

    # paginação
    df_page = df.iloc[offset: offset + limit]

    items = []
    for _, row in df_page.iterrows():
        nh = row.get("nome_hospital", "")
        out_row: Dict[str, Any] = {"nome_hospital": nh}
        for v in vars:
            raw = row.get(v, None)
            parsed = safe_parse_json(raw)
            if parsed is not None:
                if path and str(path).strip():
                    extracted = get_by_path(parsed, path)
                else:
                    unwrapped = auto_unwrap(parsed)
                    bp, bv = pick_best_path(unwrapped, max_depth=3)
                    extracted = bv if bp is not None else unwrapped
            else:
                extracted = None
            out_row[v] = summarize_value(extracted, max_chars=max_chars)
            if include_bruto:
                out_row[f"{v}__bruto"] = extracted
        items.append(out_row)

    columns = ["nome_hospital"] + vars + ([f"{v}__bruto" for v in vars] if include_bruto else [])
    return {
        "total": len(items),
        "columns": columns,
        "items": items,
        "page": {"offset": offset, "limit": limit, "total_available": total_available}
    }


@app.get("/extrair_metricas")
def extrair_metricas(
    var: str = Query(..., description="Nome da coluna/variável onde buscar (ex.: Metas)"),
    keywords: str = Query(..., description="Palavras-chave separadas por vírgula (ex.: cirurgia,cirurgias)"),
    path: Optional[str] = Query(None, description="Caminho pontuado no JSON (opcional)"),
    agg: str = Query("max", pattern="^(max|first|sum)$", description="Como consolidar múltiplos números por hospital"),
    window: int = Query(60, ge=10, le=400, description="Janela de busca ao redor das palavras (caracteres)"),
    limit: int = Query(200, ge=1, le=10000, description="Quantos hospitais retornar"),
    offset: int = Query(0, ge=0, description="Deslocamento para paginação"),
    include_context: bool = Query(True, description="Se True, retorna trecho de contexto/página/cláusula")
):
    """
    Para cada hospital, varre a variável 'var', extrai números próximos das 'keywords' e
    retorna um valor consolidado por hospital (max/first/sum) com contexto.
    """
    df = load_df()
    total_available = len(df)
    if var not in df.columns:
        return {
            "total": 0, "items": [],
            "error": f"Variável '{var}' não existe.",
            "page": {"offset": offset, "limit": limit, "total_available": total_available}
        }

    kws = [k.strip() for k in keywords.split(",") if k.strip()]
    df_page = df.iloc[offset: offset + limit]

    items = []
    for _, row in df_page.iterrows():
        nh = row.get("nome_hospital", "")
        raw = row.get(var, None)
        parsed = safe_parse_json(raw)

        if parsed is not None:
            if path and str(path).strip():
                extracted = get_by_path(parsed, path)
            else:
                unwrapped = auto_unwrap(parsed)
                bp, bv = pick_best_path(unwrapped, max_depth=3)
                extracted = bv if bp is not None else unwrapped
        else:
            extracted = None

        # transforma em lista de itens textuais
        text_items = flatten_text_items(extracted)
        all_nums: List[int] = []
        best_context = None

        for it in text_items:
            texto = it.get("texto", "")
            if not texto.strip():
                continue
            matches = find_numbers_near_keywords(texto, kws, window=window)
            if matches:
                # guarda números e (primeiro) contexto encontrado
                all_nums.extend([m["num"] for m in matches])
                if best_context is None:
                    best_context = {
                        "trecho": matches[0]["contexto"],
                        "pagina": it.get("pagina"),
                        "clausula": it.get("clausula")
                    }

        if not all_nums:
            result_val = None
        else:
            if agg == "max":
                result_val = max(all_nums)
            elif agg == "sum":
                result_val = sum(all_nums)
            else:  # first
                result_val = all_nums[0]

        out = {"nome_hospital": nh, "valor": result_val}
        if include_context:
            out["contexto"] = best_context
        items.append(out)

    return {
        "total": len(items),
        "items": items,
        "page": {"offset": offset, "limit": limit, "total_available": total_available}
    }
