"""Raccord de blocs qui se chevauchent : retire du début du nouveau texte les mots déjà
présents à la fin du texte précédent (le recouvrement audio les transcrit deux fois)."""
import re


def _norm(w):
    return re.sub(r"[^\w]", "", w.lower())


def merge_overlap(prev: str, new: str, max_words: int = 6) -> str:
    """Renvoie `new` sans son préfixe déjà contenu dans la fin de `prev`.
    Heuristique textuelle : si le moteur transcrit autrement le mot de la jointure, le doublon reste."""
    p, n = prev.split(), new.split()
    for k in range(min(max_words, len(p), len(n)), 0, -1):
        if [_norm(w) for w in p[-k:]] == [_norm(w) for w in n[:k]]:
            return " ".join(n[k:])
    return new
