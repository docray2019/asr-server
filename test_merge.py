from merge import merge_overlap


def test_doublon_retire():
    assert merge_overlap("le patient présente une gonarthrose", "gonarthrose droite évoluée") == "droite évoluée"


def test_plusieurs_mots_et_ponctuation():
    assert merge_overlap("prothèse totale de hanche,", "de hanche, gauche") == "gauche"


def test_pas_de_doublon():
    assert merge_overlap("bonjour docteur", "le genou gauche") == "le genou gauche"


def test_texte_vide():
    assert merge_overlap("", "début") == "début"
    assert merge_overlap("fin", "") == ""
