import os
import sys
import re
import pandas as pd
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

train_path = "data/processed/apple_support/splits/train.csv"
df_train = pd.read_csv(train_path, low_memory=False)

# Compile regexes with word boundaries and phrase matching
PATTERNS = {
    # 1. Keyboard & Typing Glitch (highly specific iOS 11 bug and keyboard issues)
    'keyboard_typing': re.compile(
        r'\b(?:question\s*mark(?:\s*box|\s*in\s*a\s*box)?|'
        r'autocorrect|auto-correct|auto\s*correct|'
        r'keyboard|predictive\s*text|typing|letter\s*i\b|capital\s*i\b|'
        r'symbol\s*box|strange\s*symbol|weird\s*symbol|'
        r'type\s*i\b|types\s*i\b|typing\s*i\b|i\s*turns\s*into)\b',
        re.IGNORECASE
    ),
    # 2. Battery & Power
    'battery_power': re.compile(
        r'\b(?:battery(?:\s*life|\s*drain|\s*percentage|\s*health|\s*dying|\s*dead)?|'
        r'drain|draining|drained|drains|'
        r'charge|charging|charger|cables?|lightning\s*cable|'
        r'dies\s*(?:fast|quickly|randomly)|dying\s*(?:fast|quickly)|'
        r'shutting\s*down|turns?\s*off\s*at\s*\d+%?|power\s*(?:off|down)?)\b',
        re.IGNORECASE
    ),
    # 3. Account, Apple ID & Billing
    'account_billing': re.compile(
        r'\b(?:apple\s*id|icloud(?:\s*drive|\s*storage|\s*backup|\s*photo)?|'
        r'passwords?|passcodes?|two-factor|2-factor|verification\s*code|'
        r'log\s*in|logged\s*out|sign\s*in|signing\s*in|login|'
        r'app\s*store|itunes\s*store|itunes\s*card|gift\s*card|'
        r'subscriptions?|billing|billed|charges?|charged|refunds?|'
        r'receipts?|purchases?|purchased|payment\s*method)\b',
        re.IGNORECASE
    ),
    # 4. Connectivity & Network
    'connectivity_network': re.compile(
        r'\b(?:wi-?fi|bluetooth|cellular|lte|4g|3g|mobile\s*data|'
        r'no\s*service|searching\.\.\.|carrier|hotspot|airdrop|'
        r'pair|pairing|unpair|airplay|airplane\s*mode|'
        r'sim\s*card|call\s*drops?|dropped\s*calls?|gps|location\s*services)\b',
        re.IGNORECASE
    ),
    # 5. Hardware, Audio & Display
    'hardware_audio_display': re.compile(
        r'\b(?:screens?|touchscreen|display|glass|cracks?|cracked|shattered|'
        r'black\s*screen|lines\s*on\s*screen|green\s*line|'
        r'cameras?|photos?|flash|flashlight|'
        r'audio|speakers?|microphones?|mics?|sound|volume|'
        r'earpods?|airpods?|headphones?|headphone\s*jack|dongle|'
        r'home\s*button|power\s*button|volume\s*button|vibrate|vibration)\b',
        re.IGNORECASE
    ),
    # 6. Performance, System Freezing & App Crashes
    'performance_system': re.compile(
        r'\b(?:freez(?:e|ing|es|ed)|frozen|'
        r'crash(?:es|ing|ed)?|'
        r'lag(?:ging|s)?|slow|sluggish|unresponsive|stuck|'
        r'apps?\s*(?:keep\s*closing|won\'t\s*open|quits?)|'
        r'glitch(?:es|y)?|bugs?|spinning\s*wheel|black\s*spinning)\b',
        re.IGNORECASE
    ),
    # 7. Software Update & Installation
    'software_update': re.compile(
        r'\b(?:updates?|updating|updated|'
        r'install(?:ing|ed|s)?|installation|'
        r'downloads?|downloading|downloaded|'
        r'ios\s*11(?:\.\d+)*|ios\s*10(?:\.\d+)*|high\s*sierra|'
        r'itunes(?:\s*update|\s*restore|\s*error)?|restore|restoring|'
        r'firmware|software\s*update)\b',
        re.IGNORECASE
    ),
}

def assign_intent(text):
    t = str(text).lower()
    
    # Check in order of specificity
    if PATTERNS['keyboard_typing'].search(t):
        return 'keyboard_typing'
    if PATTERNS['battery_power'].search(t):
        return 'battery_power'
    if PATTERNS['account_billing'].search(t):
        return 'account_billing'
    if PATTERNS['connectivity_network'].search(t):
        return 'connectivity_network'
    if PATTERNS['hardware_audio_display'].search(t):
        return 'hardware_audio_display'
    if PATTERNS['performance_system'].search(t):
        return 'performance_system'
    if PATTERNS['software_update'].search(t):
        return 'software_update'
    
    return 'general_inquiry_other'

df_train['intent'] = df_train['customer_text_normalized'].apply(assign_intent)

print("Intent Distribution in Training Data (73,700 queries):")
dist = df_train['intent'].value_counts()
for intent, cnt in dist.items():
    print(f"  {intent:25s}: {cnt:,} ({cnt/len(df_train)*100:.2f}%)")

print("\nSample queries per intent:")
for intent in dist.index:
    print(f"\n=== Intent: {intent} ===")
    samples = df_train[df_train['intent'] == intent]['customer_text'].head(3).tolist()
    for s in samples:
        print("  *", s.replace('\n', ' ')[:110])
