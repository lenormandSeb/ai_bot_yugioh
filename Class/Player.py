class Player:
    name = ''
    lifePoint = 0
    deck = 0

    def __init__(self, name: str, lifePoint: int=None, deck: int=None) -> None:
        self.name = name
        self.lifePoint = lifePoint
        self.deck = deck
