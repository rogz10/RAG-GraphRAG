from eval.corpus import letter_proportion, usable_chunk


def test_letter_proportion():
    test_1="06:00 145,2 152,8\n" * 20
    test_2="combien de caractères, combien de lettres,"
    test_3="La centrale de Bois-Rouge sera convertie à la biomasse"*5

    assert usable_chunk(test_1) is False
    assert usable_chunk(test_2) is False
    assert usable_chunk(test_3) is True
    ## Test letter_proportion
    assert letter_proportion(test_1) == 0
    assert letter_proportion(test_2)>= 0.5
    assert letter_proportion(test_3) >= 0.5
