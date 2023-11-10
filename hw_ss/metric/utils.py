# Don't forget to support cases when target_text == ''
from editdistance import eval

def calc_cer(target_text: str, predicted_text: str) -> float:
    target_length = len(target_text)
    if target_length == 0: return int(bool(predicted_text))
    return eval(target_text, predicted_text) / target_length


def calc_wer(target_text: str, predicted_text: str) -> float:
    # WER = normalized levenstein
    target_length = len(target_text)
    if target_length == 0: return int(bool(predicted_text)) 
    return eval(target_text.split(), predicted_text.split()) / target_length