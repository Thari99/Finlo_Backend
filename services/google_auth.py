"""
Verifies Google ID tokens received from the Flutter app.

The app obtains the ID token via `google_sign_in`. Backend uses Google's
public certificates (fetched + cached by `google-auth`) to verify the
signature and validate the `aud` claim against our registered Web OAuth
client ID.

Returns a dict with the verified `sub` (stable Google user id), `email`,
and `name`. Caller mints a backend JWT from this.
"""
from fastapi import HTTPException
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from config import GOOGLE_CLIENT_ID

_request = google_requests.Request()


def verify_google_id_token(token: str) -> dict:
    try:
        info = google_id_token.verify_oauth2_token(
            token,
            _request,
            GOOGLE_CLIENT_ID,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid Google ID token: {e}",
        ) from e

    sub = info.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Google token missing 'sub' claim")

    return {
        "sub": sub,
        "email": info.get("email", ""),
        "name": info.get("name", ""),
        "picture": info.get("picture", ""),
        "email_verified": info.get("email_verified", False),
    }
