import os
import sys
import re
import json
import pandas as pd
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')
train_path = "data/processed/apple_support/splits/train.csv"
df_train = pd.read_csv(train_path, low_memory=False)

texts = df_train['customer_text_normalized'].fillna('').str.lower()

# Domain keyword dictionaries
domain_patterns = {
    'Battery & Power': re.compile(r'\b(?:battery|drain|draining|drains|charge|charging|charger|dies|dying|power)\b', re.I),
    'Software Update & Installation': re.compile(r'\b(?:update|updating|updated|install|installing|downloading|restore|restoring|backup|ios\s*11|high sierra|itunes update)\b', re.I),
    'App & System Performance': re.compile(r'\b(?:freeze|freezing|frozen|crash|crashing|crashes|lag|lagging|slow|sluggish|unresponsive|stuck|glitch|bug)\b', re.I),
    'Keyboard & Text Input Glitch': re.compile(r'\b(?:keyboard|autocorrect|type|typing|letter|question mark|question mark box|capital i|predictive)\b', re.I),
    'Apple ID, Account & Billing': re.compile(r'\b(?:apple\s*id|icloud|password|passcode|account|login|logged|log in|store|app\s*store|subscription|billing|billed|charge|charged|refund|receipt|purchase)\b', re.I),
    'Hardware, Audio & Display': re.compile(r'\b(?:screen|display|touchscreen|crack|cracked|shattered|camera|audio|speaker|mic|microphone|volume|sound|headphones|earpods|airpods|headphone jack)\b', re.I),
    'Connectivity & Network': re.compile(r'\b(?:wifi|wi-fi|bluetooth|cellular|data|signal|no service|carrier|airdrop|hotspot|connecting|pair|pairing)\b', re.I),
    'Clarification / Courtesy Follow-up': re.compile(r'^(?:@\w+\s+)*(?:yes|no|done|thanks|thank you|ok|okay|already|yep|nope|ios\s*\d+(?:\.\d+)*|\d+(?:\.\d+)+|iphone\s*(?:\d+[a-z]*|x|se)|ipad|macbook|watch)\s*$', re.I),
}

matches = {k: texts.apply(lambda s: bool(pat.search(s))).sum() for k, pat in domain_patterns.items()}

print("Matches per domain in Training Data (73,700 queries):")
for k, cnt in sorted(matches.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:36s}: {cnt:,} ({cnt/len(df_train)*100:.2f}%)")
