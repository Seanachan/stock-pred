from enum import IntEnum


class TradeStatus(IntEnum):
    SELL = 0
    HOLD = 1
    BUY = 2

    def __str__(self):
        match self:
            case TradeStatus.SELL:
                return "SELL"
            case TradeStatus.HOLD:
                return "HOLD"
            case TradeStatus.BUY:
                return "BUY"
        return self.name
