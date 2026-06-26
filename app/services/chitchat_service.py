import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class ChitchatResult:
    is_chitchat: bool
    response: Optional[str] = None

CHITCHAT_PATTERNS = [
    (r"^\s*(bonjour|bonsoir|salut|hello|hi|salam|مرحبا|أهلا)\s*[!.?]*\s*$",
     "Bonjour ! Je suis l'assistant juridique ANP Legal. Comment puis-je vous aider avec vos dossiers ?"),
    (r"^\s*(merci|shukran|شكرا|thank you|thanks)\s*[!.]*\s*$",
     "De rien ! N'hésitez pas si vous avez d'autres questions sur vos dossiers."),
    (r"^\s*(ok|okay|d'accord|bien|parfait|très bien|compris|vu)\s*[!.]*\s*$",
     "Très bien ! Y a-t-il autre chose que je peux faire pour vous ?"),
    (r"^\s*(au revoir|bye|bonne journée|à bientôt|bonne soirée)\s*[!.]*\s*$",
     "Au revoir ! Bonne journée."),
    (r"(qui es.tu|tu es quoi|c'est quoi|présente.toi|que fais.tu)",
     "Je suis l'assistant juridique ANP Legal. Je peux répondre à vos questions sur vos dossiers, honoraires, séances et documents juridiques."),
    (r"^\s*(test|ping|allo|allô)\s*[!.?]*\s*$",
     "Je suis opérationnel ! Posez-moi une question sur vos dossiers juridiques."),
]

def detect_chitchat(question: str) -> ChitchatResult:
    q = question.strip().lower()
    for pattern, response in CHITCHAT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return ChitchatResult(is_chitchat=True, response=response)
    words = [w for w in q.split() if len(w) > 2]
    legal_keywords = {"dossier", "honoraire", "séance", "avocat", "jugement",
                      "document", "client", "statut", "date", "montant", "affaire"}
    if len(words) <= 1 and not any(kw in q for kw in legal_keywords):
        return ChitchatResult(
            is_chitchat=True,
            response="Pouvez-vous préciser votre question ? Je peux vous aider avec vos dossiers juridiques, honoraires, séances et documents."
        )
    return ChitchatResult(is_chitchat=False)