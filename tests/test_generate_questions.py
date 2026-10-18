from eval.generate_questions import copy_ratio, vague_reference


def test_vague_reference():
    assert vague_reference.search("quel est le rôle de cette commission locale ?")
    assert not vague_reference.search("qu'est-ce qu'une zone non interconnectée ?")
def test_copy_ratio():
    passage = "Le tarif réduit est accordé aux voyageurs de moins de 26 ans."
    assert copy_ratio("Quel tarif réduit est accordé aux voyageurs ?", passage) == 0.8
    assert copy_ratio("Jusqu'à quel âge paie-t-on moins cher ?", passage) < 0.5
