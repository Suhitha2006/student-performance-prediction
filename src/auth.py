import hashlib

# Pre-configured demonstration accounts matching both email and username formats
USERS = {
    "admin@edupredict.org": {
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "Administrator",
        "name": "Prof. Suhitha (Admin)",
        "email": "admin@edupredict.org"
    },
    "teacher@edupredict.org": {
        "username": "teacher",
        "password_hash": hashlib.sha256("teacher123".encode()).hexdigest(),
        "role": "Academic Faculty",
        "name": "Sarah Jenkins (Faculty)",
        "email": "teacher@edupredict.org"
    },
    "student@edupredict.org": {
        "username": "student",
        "password_hash": hashlib.sha256("student123".encode()).hexdigest(),
        "role": "Student Learner",
        "name": "Alex Morgan (Student)",
        "email": "student@edupredict.org"
    }
}

# Also support plain usernames
for k, v in list(USERS.items()):
    USERS[v["username"]] = v

def verify_credentials(identifier, password):
    """Verifies username or email against secure hashes."""
    identifier = identifier.strip().lower()
    if identifier in USERS:
        candidate_hash = hashlib.sha256(password.encode()).hexdigest()
        if candidate_hash == USERS[identifier]["password_hash"]:
            return USERS[identifier]
    return None
