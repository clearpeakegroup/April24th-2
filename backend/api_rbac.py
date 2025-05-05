import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret")
if SECRET_KEY == "dev-secret":
    import warnings
    warnings.warn("Using default JWT secret key! Set JWT_SECRET_KEY in production.")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

ROLES = {
    "admin": ["upload", "train", "backtest", "forwardtest", "live"],
    "trader": ["train", "backtest", "forwardtest", "live"],
    "viewer": ["backtest", "forwardtest"]
}

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

def require_role(role: str):
    def role_checker(user=Depends(get_current_user)):
        if user.get("role") not in ROLES or role not in ROLES[user["role"]]:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker 