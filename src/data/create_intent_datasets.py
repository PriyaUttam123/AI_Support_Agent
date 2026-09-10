"""Create intent-labeled datasets maintaining strict conversation split isolation."""

import os
import re
import json
import pandas as pd
import numpy as np

SPLITS_DIR = os.path.join("data", "processed", "apple_support", "splits")
INTENT_DIR = os.path.join("data", "processed", "apple_support", "intent")
CONFIGS_DIR = os.path.join("configs")

# Compile domain patterns in priority order
PATTERNS = {
    'keyboard_typing': re.compile(
        r'\b(?:question\s*mark(?:\s*box|\s*in\s*a\s*box)?|'
        r'autocorrect|auto-correct|auto\s*correct|'
        r'keyboard|predictive\s*text|typing|letter\s*i\b|capital\s*i\b|'
        r'symbol\s*box|strange\s*symbol|weird\s*symbol|'
        r'type\s*i\b|types\s*i\b|typing\s*i\b|i\s*turns\s*into)\b',
        re.IGNORECASE
    ),
    'battery_power': re.compile(
        r'\b(?:battery(?:\s*life|\s*drain|\s*percentage|\s*health|\s*dying|\s*dead)?|'
        r'drain|draining|drained|drains|'
        r'charge|charging|charger|cables?|lightning\s*cable|'
        r'dies\s*(?:fast|quickly|randomly)|dying\s*(?:fast|quickly)|'
        r'shutting\s*down|turns?\s*off\s*at\s*\d+%?|power\s*(?:off|down)?)\b',
        re.IGNORECASE
    ),
    'account_billing': re.compile(
        r'\b(?:apple\s*id|icloud(?:\s*drive|\s*storage|\s*backup|\s*photo)?|'
        r'passwords?|passcodes?|two-factor|2-factor|verification\s*code|'
        r'log\s*in|logged\s*out|sign\s*in|signing\s*in|login|'
        r'app\s*store|itunes\s*store|itunes\s*card|gift\s*card|'
        r'subscriptions?|billing|billed|charges?|charged|refunds?|'
        r'receipts?|purchases?|purchased|payment\s*method)\b',
        re.IGNORECASE
    ),
    'connectivity_network': re.compile(
        r'\b(?:wi-?fi|bluetooth|cellular|lte|4g|3g|mobile\s*data|'
        r'no\s*service|searching\.\.\.|carrier|hotspot|airdrop|'
        r'pair|pairing|unpair|airplay|airplane\s*mode|'
        r'sim\s*card|call\s*drops?|dropped\s*calls?|gps|location\s*services)\b',
        re.IGNORECASE
    ),
    'hardware_audio_display': re.compile(
        r'\b(?:screens?|touchscreen|display|glass|cracks?|cracked|shattered|'
        r'black\s*screen|lines\s*on\s*screen|green\s*line|'
        r'cameras?|photos?|flash|flashlight|'
        r'audio|speakers?|microphones?|mics?|sound|volume|'
        r'earpods?|airpods?|headphones?|headphone\s*jack|dongle|'
        r'home\s*button|power\s*button|volume\s*button|vibrate|vibration)\b',
        re.IGNORECASE
    ),
    'performance_system': re.compile(
        r'\b(?:freez(?:e|ing|es|ed)|frozen|'
        r'crash(?:es|ing|ed)?|'
        r'lag(?:ging|s)?|slow|sluggish|unresponsive|stuck|'
        r'apps?\s*(?:keep\s*closing|won\'t\s*open|quits?)|'
        r'glitch(?:es|y)?|bugs?|spinning\s*wheel|black\s*spinning)\b',
        re.IGNORECASE
    ),
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

INTENT_TAXONOMY = {
    'battery_power': {
        'description': 'Battery drain, rapid discharge, power management, charging failure, or unexpected shutdowns.',
        'in_scope': 'Battery health, charging cables, fast drain, phone dying above 1%, overheating while charging.',
        'out_of_scope': 'General phone slowness or app freezing without battery mention.',
        'confusable_intents': ['performance_system', 'software_update'],
        'decision_boundary': 'Classify as battery_power if battery life, drain, or charging is explicitly mentioned.'
    },
    'software_update': {
        'description': 'Issues installing, downloading, verifying, or rolling back iOS/macOS updates or iTunes backup/restore.',
        'in_scope': 'Update verification hang, installation errors, iTunes update errors, release inquiries, reverting update.',
        'out_of_scope': 'Specific bugs caused by an update (e.g. battery drain or keyboard glitch).',
        'confusable_intents': ['battery_power', 'performance_system', 'keyboard_typing'],
        'decision_boundary': 'If customer asks how/when to update, or update failed to install/download, classify as software_update.'
    },
    'performance_system': {
        'description': 'System sluggishness, freezing touchscreen, app crashes, unexpected lag, or unresponsiveness.',
        'in_scope': 'Apps closing spontaneously, touchscreen freezing, slow typing/navigation, spinning wheel.',
        'out_of_scope': 'Physical screen cracking, pure battery drain, or keyboard autocorrect letter substitution.',
        'confusable_intents': ['software_update', 'hardware_audio_display', 'keyboard_typing'],
        'decision_boundary': 'Classify as performance_system when the primary complaint is software unresponsiveness, freezing, or app crashes.'
    },
    'keyboard_typing': {
        'description': 'Autocorrect bugs (letter i becoming question mark box), keyboard lag, predictive text or dictation failures.',
        'in_scope': 'Question mark box bug, keyboard freezing/lag, predictive emoji issues, autocorrect capitalization glitches.',
        'out_of_scope': 'General app freezing outside of text entry.',
        'confusable_intents': ['performance_system', 'software_update'],
        'decision_boundary': 'Specific to text entry, keyboard responsiveness, predictive text, and autocorrect anomalies.'
    },
    'account_billing': {
        'description': 'Apple ID credentials, iCloud storage/sync, passwords, App Store purchases, and subscription charges.',
        'in_scope': 'Password resets, 2FA codes, account lockout, subscription cancellations, refund requests, iCloud full warnings.',
        'out_of_scope': 'Hardware or device connectivity issues.',
        'confusable_intents': ['general_inquiry_other'],
        'decision_boundary': 'Involves personal credentials, Apple ID authentication, or financial/subscription transactions.'
    },
    'hardware_audio_display': {
        'description': 'Physical hardware damage, cracked display, black screen, camera failure, microphone, speaker, or headphone issues.',
        'in_scope': 'Broken glass, OLED lines, camera black screen, distorted speaker sound, mic not working, headphone dongle defect.',
        'out_of_scope': 'Software touchscreen lag without physical damage.',
        'confusable_intents': ['performance_system'],
        'decision_boundary': 'Classify here if physical components (screen, camera, speaker, mic, buttons, audio ports) fail or are damaged.'
    },
    'connectivity_network': {
        'description': 'Wi-Fi disconnects, Bluetooth pairing failures, cellular signal drops, No Service errors, or carrier data.',
        'in_scope': 'Cannot join Wi-Fi, grayed out toggle, Bluetooth dropouts, dropped cellular calls, No Service, Hotspot.',
        'out_of_scope': 'App Store download speed issues without network disconnection.',
        'confusable_intents': ['hardware_audio_display', 'general_inquiry_other'],
        'decision_boundary': 'Involves wireless protocols (Wi-Fi, Bluetooth, Cellular, AirDrop) or cellular carrier network access.'
    },
    'general_inquiry_other': {
        'description': 'General inquiries, store hours, reservations, courtesy messages, mid-thread confirmations, or conversational follow-ups.',
        'in_scope': 'Pre-order inquiries, store location questions, courtesy ("thanks", "done"), short answers ("yes", "iPhone 7"), unspecific complaints.',
        'out_of_scope': 'Inquiries with specific technical or account failure symptoms.',
        'confusable_intents': ['account_billing', 'software_update'],
        'decision_boundary': 'Fallback class for generic inquiries, social acknowledgements, or messages lacking specific technical domain symptoms.'
    }
}


def label_intent(text: str) -> str:
    """Assign ground-truth intent using priority-ordered domain regexes."""
    t = str(text).lower()
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


def create_intent_datasets():
    os.makedirs(INTENT_DIR, exist_ok=True)
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    print("=" * 70)
    print("PHASE 5: Generating Intent-Labeled Datasets")
    print("=" * 70)

    # Save intent taxonomy configuration
    taxonomy_file = os.path.join(CONFIGS_DIR, "intent_taxonomy.json")
    with open(taxonomy_file, "w", encoding="utf-8") as f:
        json.dump(INTENT_TAXONOMY, f, indent=2)
    print(f"Saved intent taxonomy configuration to {taxonomy_file}")

    split_files = {
        'train': os.path.join(SPLITS_DIR, "train.csv"),
        'validation': os.path.join(SPLITS_DIR, "validation.csv"),
        'test': os.path.join(SPLITS_DIR, "test.csv"),
    }

    intent_distributions = {}

    for split_name, input_path in split_files.items():
        print(f"\nProcessing {split_name} split from {input_path}...")
        df = pd.read_csv(input_path, low_memory=False)
        
        # Assign intent
        df['intent'] = df['customer_text_normalized'].apply(label_intent)
        
        # Save labeled dataset
        out_path = os.path.join(INTENT_DIR, f"{split_name}_intents.csv")
        df.to_csv(out_path, index=False, encoding='utf-8')
        print(f"  -> Saved {len(df):,} labeled records to {out_path}")
        
        dist = df['intent'].value_counts().to_dict()
        intent_distributions[split_name] = {
            'total': len(df),
            'distribution': dist,
            'percentages': {k: round(v / len(df) * 100, 2) for k, v in dist.items()}
        }

    # Save summary report JSON
    stats_file = os.path.join("reports", "intent_distribution_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(intent_distributions, f, indent=2)
    print(f"\nSaved intent distribution summary to {stats_file}")

    return intent_distributions


if __name__ == "__main__":
    create_intent_datasets()
