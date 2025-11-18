import re

def getFreq_in_ms(frequency: str) -> int:
    # Regular expression to match any of the characters 'm', 'H', 'D', 'W', 'M'
    freq = int(re.split(r"[mHDWM]", frequency, maxsplit=1)[0])
    if frequency.find('m') != -1:
        return freq * 60 * 1000
    if frequency.find('H') != -1:
        return freq * 60 * 60 * 1000
    if frequency.find('D') != -1:
        return freq * 24 * 60 * 60 * 1000
    if frequency.find('W') != -1:
        return freq * 7 * 24 * 60 * 60 * 1000
    if frequency.find('M') != -1:
        return freq * 31 * 24 * 60 * 60 * 1000
    return 0

def freq_to_resample(f: str) -> str:
    m = f.strip().lower()
    return {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1H"
    }.get(m, "3min")

def resample_to_freq(frequency: str) -> str:
    # Replace characters
    if frequency.find('min') != -1:
        return frequency.replace("min", "m").lower()
    else:
        return frequency