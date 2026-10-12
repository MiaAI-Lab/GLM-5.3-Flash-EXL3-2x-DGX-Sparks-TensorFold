#!/usr/bin/env python3
"""End of turn on short French coding prompts (thinking off): replies that run to max_tokens instead of ending their
turn, and P(end of turn) right after the reply's closing code fence (one token drawn at T=1, top_p=1, over 40
seeds, after the prompt plus the T=0 reply up to its fence, by token ids).

Usage: tools/end_of_turn.py [label] [max_cut]      (API_URL / PORT as in client.py). Exit code 1 when more than
max_cut replies (default 4) of the 48 are cut.

On 2x GB10: DENSE=bf16 1 of 48 cut, P 0.76; fp8 1, 0.82; q4 12, 0.55 (issue #18).
"""
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True           # no tools/__pycache__ from importing client
from client import API_URL, open_url  # noqa: E402

TASKS = [
    "Écris une fonction Python `fusionne_tries(listes)` qui fusionne une liste de listes triées en une seule liste triée, en O(N log k) avec heapq. Réponds uniquement avec le code.",
    "Écris une classe Python `LRU(capacite)` avec `get(cle)` (renvoie -1 si absente) et `put(cle, valeur)`, éviction du moins récemment utilisé, O(1). Réponds uniquement avec le code.",
    "Écris une fonction Python `vers_romain(n)` (1 ≤ n ≤ 3999) qui renvoie le nombre en chiffres romains. Réponds uniquement avec le code.",
    "Écris une fonction Python `equilibre(s)` qui renvoie True si les parenthèses (), [], {} de la chaîne sont correctement imbriquées. Réponds uniquement avec le code.",
    "Écris une fonction Python `tri_topologique(n, aretes)` qui renvoie un ordre topologique des sommets 0..n-1 (aretes = liste de (u,v) signifiant u avant v), ou None s'il y a un cycle. Réponds uniquement avec le code.",
    "Écris une fonction Python `spirale(m)` qui renvoie les éléments d'une matrice (liste de listes) parcourue en spirale dans le sens horaire depuis le coin haut-gauche. Réponds uniquement avec le code.",
    "Écris deux fonctions Python `rle_encode(s)` et `rle_decode(s)` : encodage par plages sous la forme '3a2b1c' pour 'aaabbc' (compte puis caractère, pas de chiffres dans l'entrée), et l'inverse. Réponds uniquement avec le code.",
    "Écris une fonction Python `duree_secondes(s)` qui convertit une durée ISO 8601 de la forme PnDTnHnMnS (chaque composante optionnelle, entiers) en secondes. Réponds uniquement avec le code.",
]
OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(API_URL + path, json.dumps({"model": "GLM-5.3-Flash-EXL3", **body}).encode(),
                                 {"Content-Type": "application/json"})
    with open_url(req, timeout=900) as r:
        return json.load(r)


def cut_replies() -> tuple[int, int]:
    """Replies ended by max_tokens: each task 5 times at T=0.7 (seeds 1000-1004) and once at T=0."""

    jobs = [(p, 1000 + i, 0.7) for p in TASKS for i in range(5)] + [(p, None, 0.0) for p in TASKS]

    def one(job) -> bool:
        p, seed, temp = job
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": 1024, "temperature": temp, **OFF}
        if seed is not None:
            body["seed"] = seed
        return post("/v1/chat/completions", body)["choices"][0]["finish_reason"] == "length"

    with ThreadPoolExecutor(4) as ex:
        cut = list(ex.map(one, jobs))
    return sum(cut), len(cut)


def p_end(prompt: str, draws: int = 40) -> float | None:
    """P(end of turn) right after the closing fence of the T=0 reply, or None when the reply has no closed fence."""

    msgs = [{"role": "user", "content": prompt}]
    reply = post("/v1/chat/completions", {"messages": msgs, "max_tokens": 1024, "temperature": 0,
                                          "return_token_ids": True, **OFF})
    gen = (reply.get("tensorfold") or {}).get("token_ids") or reply["choices"][0].get("token_ids") or []
    text, cut = "", None
    for k, t in enumerate(gen):
        text += post("/detokenize", {"tokens": [t]}).get("prompt", "")
        if text.count("```") >= 2:
            cut = k + 1
            break
    if cut is None:
        return None
    ids = post("/tokenize", {"messages": msgs, **OFF})["tokens"] + gen[:cut]
    stops = 0
    for seed in range(draws):
        c = post("/v1/completions", {"prompt": ids, "max_tokens": 1, "temperature": 1.0, "top_p": 1.0,
                                     "seed": seed})["choices"][0]
        stops += c["finish_reason"] == "stop" and not c["text"]
    return stops / draws


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else API_URL
    max_cut = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n, total = cut_replies()
    ps = [p for p in map(p_end, TASKS) if p is not None]
    mean = f"mean {sum(ps) / len(ps):.2f}, min {min(ps):.2f} ({len(ps)} tasks)" if ps else "no closed fence"
    print(f"{label}: replies cut by max_tokens {n}/{total}; P(end of turn) after the closing fence: {mean}")
    sys.exit(1 if n > max_cut else 0)


if __name__ == "__main__":
    main()
