from scripts.lacuna_matched002 import select_history

def test_history_selection_uses_observed_context_and_age_only():
 h={b'abcd':[20,40,80]}
 assert select_history(h,b'abcd',100,8,32)==80
 assert select_history(h,b'abcd',100,32,128)==40
 assert select_history(h,b'other',100,8,128) is None
 assert select_history(h,b'abcd',10,8,128) is None
