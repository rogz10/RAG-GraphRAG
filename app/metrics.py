from prometheus_client import Counter, Histogram

REQUESTS = Counter(
    "rag_requests_total",
    "nombre de requêtes reçues par l'api",
    ["endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "durée de traitement d'une requête",
    ["endpoint"],
)
