import os
import sys
import re
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
train_path = "data/processed/apple_support/splits/train.csv"
df_train = pd.read_csv(train_path, low_memory=False)

texts = df_train['customer_text_normalized'].fillna('').str.lower()

domain_patterns = {
    'battery_power': re.compile(r'\b(?:battery|drain|draining|drains|charge|charging|charger|dies|dying|power)\b', re.I),
    'software_update': re.compile(r'\b(?:update|updating|updated|install|installing|downloading|restore|restoring|backup|ios\s*11|high sierra|itunes update)\b', re.I),
    'performance_freeze': re.compile(r'\b(?:freeze|freezing|frozen|crash|crashing|crashes|lag|lagging|slow|sluggish|unresponsive|stuck|glitch|bug)\b', re.I),
    'keyboard_typing': re.compile(r'\b(?:keyboard|autocorrect|type|typing|letter|question mark|question mark box|capital i|predictive)\b', re.I),
    'account_billing': re.compile(r'\b(?:apple\s*id|icloud|password|passcode|account|login|logged|log in|store|app\s*store|subscription|billing|billed|charge|charged|refund|receipt|purchase)\b', re.I),
    'hardware_display': re.compile(r'\b(?:screen|display|touchscreen|crack|cracked|shattered|camera|audio|speaker|mic|microphone|volume|sound|headphones|earpods|airpods|headphone jack)\b', re.I),
    'connectivity_network': re.compile(r'\b(?:wifi|wi-fi|bluetooth|cellular|data|signal|no service|carrier|airdrop|hotspot|connecting|pair|pairing)\b', re.I),
}

# Check how many queries match 0 patterns
matched_any = texts.apply(lambda s: any(pat.search(s) for pat in domain_patterns.values()))
print(f"Total training queries: {len(df_train):,}")
print(f"Matched at least 1 technical domain: {matched_any.sum():,} ({matched_any.sum()/len(df_train)*100:.2f}%)")
print(f"Unmatched queries: {(~matched_any).sum():,} ({(~matched_any).sum()/len(df_train)*100:.2f}%)")

print("\nSample unmatched queries:")
for q in df_train[~matched_any]['customer_text'].head(15):
    print("  *", q.replace('\n', ' ')[:100])
