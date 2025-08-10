import json
import ast
import difflib
import pandas as pd
from fastapi import FastAPI, Request
from unidecode import unidecode

# === ajuste se o arquivo tiver outro nome/ caminho ===
DB_PATH = "hospitais_BD_final.csv"

_df_cache = None

def load_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(DB_PATH, dtype=str)
    return _df_cache

def norm(s: str) -> str:
    return unidecode(str(s or "")).strip().lower()

def safe_parse_json(texto):
    """
    Tenta json.loads; se falhar, tenta ast.literal_eval.
    Se não parseável, retorna {'_raw': texto, '_parsed': False}.
    """
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

def selecionar_por_nome(df: pd.DataFrame,
                        nome_informado: str,
                        max_sugestoes: int = 5,
                        auto_picking: bool = True,
                        score_min: float = 0.85):
    """
    Fluxo:
      1) match exato (normalizado)
      2) se não achar: 'contains' (parcial)
      3) se 'contains' tiver 1: escolhe
      4) se 'contains' tiver >1: retorna opções p/ desambiguação
      5) se 'contains' tiver 0: fuzzy (difflib) e retorna sugestões.
         Se top score >= score_min e auto_picking=True, seleciona.
    """
    if "nome_hospital" not in df.columns:
        return {"status": "nao_encontrado", "opcoes": [], "erro": "Coluna 'nome_hospital' não existe no CSV."}

    alvo = norm(nome_informado)
    nomes = df["nome_hospital"].fillna("")
    nomes_norm = nomes.map(norm)

    # 1) Exato
    sub = df[nomes_norm == alvo]
    if not sub.empty:
        return {"status": "selecionado", "row": sub.iloc[0].to_dict()}

    # 2) Contains (parcial)
    mask_contains = nomes_norm.str.contains(alvo, regex=False, na=False)
    sub_contains = df[mask_contains]
    if len(sub_contains) == 1:
        return {"status": "selecionado", "row": sub_contains.iloc[0].to_dict()}
    if len(sub_contains) > 1:
        return {"status": "desambiguar",
                "opcoes": sub_contains["nome_hospital"].head(max_sugestoes).tolist()}

    # 3) Fuzzy (aproximação)
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

app = FastAPI(title="Ações do Banco de Hospitais", version="1.0.0")

@app.post("/avaliar_variaveis_json")
async def avaliar_variaveis_json(request: Request):
    """
    Body esperado:
    {
      "selector": { "nome_hospital": "..." },
      "variaveis": ["Metas", "Pagamento", "Indicadores de Qualidade", ...]
    }
    """
    body = await request.json()
    selector = body.get("selector", {})
    variaveis = body.get("variaveis", []) or []

    df = load_df()

    # Seleção por nome (com parcial + fuzzy)
    if "nome_hospital" not in selector:
        return {"ok": False, "error": "Use selector.nome_hospital."}

    sel = selecionar_por_nome(
        df,
        selector["nome_hospital"],
        max_sugestoes=5,
        auto_picking=True,   # se quiser sempre perguntar, troque para False
        score_min=0.85       # ajuste a sensibilidade do fuzzy
    )

    if sel.get("erro"):
        return {"ok": False, "error": sel["erro"]}

    if sel["status"] == "selecionado":
        row = sel["row"]
    elif sel["status"] == "desambiguar":
        return {
            "ok": False,
            "error": "Foram encontradas múltiplas opções. Escolha uma.",
            "sugestoes": sel["opcoes"]
        }
    else:
        return {"ok": False, "error": "Hospital não encontrado.", "sugestoes": sel["opcoes"]}

    # Leitura/parse das variáveis solicitadas
    resultados = {}
    for var in variaveis:
        if var not in row:
            resultados[var] = {"_error": "Variável inexistente nesta base."}
            continue
        val = row[var]
        if val is None or str(val).strip() == "":
            resultados[var] = {"_error": "Variável vazia."}
            continue
        resultados[var] = safe_parse_json(val)

    return {
        "ok": True,
        "hospital": {"nome_hospital": row.get("nome_hospital")},
        "resultados": resultados
    }
