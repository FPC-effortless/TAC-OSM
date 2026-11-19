"""Time the selected native candidate against the frozen CPU controls."""
from benchmark import measure


if __name__ == '__main__':
    measure(('tac_selected', 'nanochat', 'gru'))
