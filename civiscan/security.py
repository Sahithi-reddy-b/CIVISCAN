"""Reusable role and account-status authorization checks."""


class AuthorizationError(PermissionError):
    """Raised when a user lacks permission for a protected operation."""


def require_role(user, allowed_roles):
    """Return an active user only when their role is explicitly allowed."""
    if not user or user.get("account_status", "active") != "active":
        raise AuthorizationError("An active authenticated account is required.")
    if user.get("role") not in set(allowed_roles):
        raise AuthorizationError("You are not authorized to perform this action.")
    return user