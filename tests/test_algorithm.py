from evq.algorithm import EVQ


def test_one_by_one():
    c = EVQ(number_of_classes=2, vigilance=0.1)
    c.partial_fit([-2, -2], 1)
    c.partial_fit([-1, -1], 0)
    c.partial_fit([1, 1], 0)
    c.partial_fit([2, 2], 1)

    assert c.predict([0, 0]) == 0
    assert c.predict([3, 3]) == 1
    assert c.predict([-3, -3]) == 1


def test_multi():
    c = EVQ(number_of_classes=2, vigilance=0.1)
    c.fit(
        [[-2, -2], [-1, -1], [1, 1], [2, 2]],
        [1, 0, 0, 1],
        epochs=1, permute=False
    )

    assert all(c.predict([[0, 0], [3, 3], [-3, -3]]) == [0, 1, 1])
