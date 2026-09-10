"""Rule-based intent classifier for AppleSupport customer inquiries."""

import re
from typing import Dict, Any, List, Union


class RuleBasedIntentClassifier:
    """Explainable, deterministic rule-based intent classifier with rule matching and confidence scoring."""

    def __init__(self):
        # Priority-ordered rule set: (intent_name, regex_pattern, rule_description, base_confidence)
        self.rules = [
            (
                'keyboard_typing',
                re.compile(r'\b(?:question\s*mark(?:\s*box|\s*in\s*a\s*box)?|letter\s*i\b|capital\s*i\b|symbol\s*box|strange\s*symbol|type\s*i\b|types\s*i\b|i\s*turns\s*into)\b', re.I),
                'rule_ios11_autocorrect_symbol_glitch',
                0.95
            ),
            (
                'keyboard_typing',
                re.compile(r'\b(?:autocorrect|auto-correct|auto\s*correct|keyboard(?:\s*lag)?|predictive\s*text|typing)\b', re.I),
                'rule_keyboard_text_input',
                0.85
            ),
            (
                'battery_power',
                re.compile(r'\b(?:battery(?:\s*life|\s*drain|\s*percentage|\s*health|\s*dying|\s*dead)?|drain|draining|drained|drains|dies\s*(?:fast|quickly|randomly)|dying\s*(?:fast|quickly))\b', re.I),
                'rule_battery_drain_and_health',
                0.90
            ),
            (
                'battery_power',
                re.compile(r'\b(?:charge|charging|charger|lightning\s*cable|shutting\s*down|turns?\s*off\s*at\s*\d+%?|power\s*(?:off|down)?)\b', re.I),
                'rule_power_and_charging',
                0.85
            ),
            (
                'account_billing',
                re.compile(r'\b(?:apple\s*id|icloud(?:\s*drive|\s*storage|\s*backup|\s*photo)?|passwords?|passcodes?|two-factor|2-factor|verification\s*code)\b', re.I),
                'rule_account_security_credentials',
                0.90
            ),
            (
                'account_billing',
                re.compile(r'\b(?:app\s*store|itunes\s*store|subscriptions?|billing|billed|charges?|charged|refunds?|receipts?|purchases?|purchased|payment\s*method)\b', re.I),
                'rule_billing_purchases_subscriptions',
                0.85
            ),
            (
                'connectivity_network',
                re.compile(r'\b(?:wi-?fi|bluetooth|cellular|lte|4g|3g|mobile\s*data|no\s*service|searching\.\.\.|carrier|hotspot|airdrop|pair|pairing|unpair|airplay|airplane\s*mode|sim\s*card)\b', re.I),
                'rule_wireless_and_carrier_network',
                0.85
            ),
            (
                'hardware_audio_display',
                re.compile(r'\b(?:screens?|touchscreen|display|glass|cracks?|cracked|shattered|black\s*screen|lines\s*on\s*screen|green\s*line)\b', re.I),
                'rule_display_and_glass_damage',
                0.85
            ),
            (
                'hardware_audio_display',
                re.compile(r'\b(?:cameras?|photos?|flash|flashlight|audio|speakers?|microphones?|mics?|sound|volume|earpods?|airpods?|headphones?|headphone\s*jack|dongle|home\s*button|power\s*button|volume\s*button|vibrate|vibration)\b', re.I),
                'rule_hardware_audio_camera_buttons',
                0.85
            ),
            (
                'performance_system',
                re.compile(r'\b(?:freez(?:e|ing|es|ed)|frozen|crash(?:es|ing|ed)?|lag(?:ging|s)?|slow|sluggish|unresponsive|stuck|apps?\s*(?:keep\s*closing|won\'t\s*open|quits?)|spinning\s*wheel)\b', re.I),
                'rule_system_freeze_crash_lag',
                0.85
            ),
            (
                'software_update',
                re.compile(r'\b(?:updates?|updating|updated|install(?:ing|ed|s)?|installation|downloads?|downloading|downloaded|ios\s*11(?:\.\d+)*|ios\s*10(?:\.\d+)*|high\s*sierra|itunes(?:\s*update|\s*restore|\s*error)?|restore|restoring|firmware|software\s*update)\b', re.I),
                'rule_software_update_installation',
                0.80
            ),
        ]

    def predict_single(self, text: str) -> Dict[str, Any]:
        """Predict intent for a single text inquiry."""
        t = str(text).lower()
        for intent, pattern, rule_name, confidence in self.rules:
            if pattern.search(t):
                return {
                    'predicted_intent': intent,
                    'confidence': confidence,
                    'matched_rule': rule_name
                }
        return {
            'predicted_intent': 'general_inquiry_other',
            'confidence': 0.50,
            'matched_rule': 'fallback_no_match'
        }

    def predict(self, texts: Union[List[str], Any]) -> List[Dict[str, Any]]:
        """Batch prediction returning structured prediction records."""
        return [self.predict_single(t) for t in texts]


if __name__ == "__main__":
    clf = RuleBasedIntentClassifier()
    test_queries = [
        "Why is my battery draining so fast after iOS 11?",
        "Every time I type letter I it turns into question mark in a box",
        "Can't remember my Apple ID password to download apps",
        "My screen is cracked and unresponsive",
        "Bluetooth won't pair with my car",
        "Calendar keeps freezing and crashing",
        "When will the new update be available?",
        "Thank you so much for your help!"
    ]
    for q in test_queries:
        res = clf.predict_single(q)
        print(f"Query: '{q}' -> Intent: {res['predicted_intent']} (Conf: {res['confidence']}, Rule: {res['matched_rule']})")
