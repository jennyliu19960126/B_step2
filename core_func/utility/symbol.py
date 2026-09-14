def add_symbol_posfix(symbol):
    if symbol[0] in ["0", "3"]:
        return symbol + ".SZ"
    elif symbol[0] == "8":
        return symbol + ".BJ"
    elif symbol[0] == "6":
        return symbol + ".SH"
    else:
        return symbol + ".error"