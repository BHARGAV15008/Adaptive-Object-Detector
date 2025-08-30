"""
Security Utilities
JWT authentication, rate limiting, and security headers
"""

import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import logging
from functools import wraps

import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import redis.asyncio as redis

from ..core.config import get_config

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
config = get_config()
JWT_SECRET = config.get('api.authentication.jwt_secret', 'default-secret-key')
JWT_ALGORITHM = config.get('api.authentication.jwt_algorithm', 'HS256')
JWT_EXPIRATION = config.get('api.authentication.jwt_expiration', 86400)


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(seconds=JWT_EXPIRATION)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


def get_current_user(token: str) -> Dict[str, Any]:
    """Get current user from token"""
    payload = verify_token(token)
    user_id = payload.get("sub")
    
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # In a real application, you would fetch user from database
    return {
        "id": int(user_id),
        "username": payload.get("username", "unknown"),
        "roles": payload.get("roles", [])
    }


class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self):
        self.config = get_config()
        self.rate_config = self.config.get('api.rate_limiting', {})
        self.enabled = self.rate_config.get('enabled', False)
        
        self.requests_per_minute = self.rate_config.get('requests_per_minute', 100)
        self.requests_per_hour = self.rate_config.get('requests_per_hour', 1000)
        
        # Redis connection
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis connection"""
        if not self.enabled:
            return
        
        redis_config = self.config.get_database_config().get('redis', {})
        
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            password=redis_config.get('password'),
            decode_responses=True
        )
    
    async def check_rate_limit(self, identifier: str, endpoint: str = "general") -> Dict[str, Any]:
        """Check if request is within rate limits"""
        if not self.enabled or not self.redis_client:
            return {"allowed": True, "remaining": self.requests_per_minute}
        
        current_time = int(time.time())
        
        # Keys for different time windows
        minute_key = f"rate_limit:{identifier}:{endpoint}:minute:{current_time // 60}"
        hour_key = f"rate_limit:{identifier}:{endpoint}:hour:{current_time // 3600}"
        
        try:
            # Check minute limit
            minute_count = await self.redis_client.get(minute_key) or 0
            minute_count = int(minute_count)
            
            # Check hour limit
            hour_count = await self.redis_client.get(hour_key) or 0
            hour_count = int(hour_count)
            
            if minute_count >= self.requests_per_minute:
                return {
                    "allowed": False,
                    "reason": "Minute rate limit exceeded",
                    "remaining": 0,
                    "reset_time": (current_time // 60 + 1) * 60
                }
            
            if hour_count >= self.requests_per_hour:
                return {
                    "allowed": False,
                    "reason": "Hour rate limit exceeded",
                    "remaining": 0,
                    "reset_time": (current_time // 3600 + 1) * 3600
                }
            
            # Increment counters
            pipe = self.redis_client.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            await pipe.execute()
            
            return {
                "allowed": True,
                "remaining": self.requests_per_minute - minute_count - 1,
                "hour_remaining": self.requests_per_hour - hour_count - 1
            }
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request if rate limiting fails
            return {"allowed": True, "remaining": self.requests_per_minute}


class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self):
        self.config = get_config()
        self.api_key_config = self.config.get('security.api_keys', {})
        self.enabled = self.api_key_config.get('enabled', False)
        
        # In production, these would be stored in a database
        self.api_keys = {
            "admin_key": {
                "user_id": 1,
                "username": "admin",
                "roles": ["admin"],
                "permissions": ["read", "write", "admin"],
                "created_at": datetime.utcnow(),
                "expires_at": None
            }
        }
    
    def generate_api_key(self, user_id: int, username: str, roles: List[str], 
                        expires_days: Optional[int] = None) -> str:
        """Generate a new API key"""
        # Generate secure random key
        api_key = secrets.token_urlsafe(32)
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        self.api_keys[api_key] = {
            "user_id": user_id,
            "username": username,
            "roles": roles,
            "permissions": self._get_permissions_for_roles(roles),
            "created_at": datetime.utcnow(),
            "expires_at": expires_at
        }
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user info"""
        if not self.enabled:
            return None
        
        if api_key not in self.api_keys:
            return None
        
        key_info = self.api_keys[api_key]
        
        # Check if expired
        if key_info["expires_at"] and datetime.utcnow() > key_info["expires_at"]:
            return None
        
        return key_info
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.api_keys:
            del self.api_keys[api_key]
            return True
        return False
    
    def _get_permissions_for_roles(self, roles: List[str]) -> List[str]:
        """Get permissions based on roles"""
        permissions = set()
        
        for role in roles:
            if role == "admin":
                permissions.update(["read", "write", "admin", "delete"])
            elif role == "user":
                permissions.update(["read", "write"])
            elif role == "viewer":
                permissions.add("read")
        
        return list(permissions)


class SecurityHeaders:
    """Security headers middleware"""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """Get security headers"""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()"
        }


class InputValidator:
    """Input validation and sanitization"""
    
    def __init__(self):
        self.config = get_config()
        self.validation_config = self.config.get('security.input_validation', {})
        
        self.max_image_size = self._parse_size(self.validation_config.get('max_image_size', '50MB'))
        self.allowed_formats = self.validation_config.get('allowed_formats', ['jpg', 'jpeg', 'png'])
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '50MB') to bytes"""
        size_str = size_str.upper().replace(' ', '')
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)  # Assume bytes
    
    def validate_image_upload(self, filename: str, file_size: int, content_type: str) -> Dict[str, Any]:
        """Validate image upload"""
        errors = []
        
        # Check file size
        if file_size > self.max_image_size:
            errors.append(f"File size ({file_size} bytes) exceeds maximum allowed ({self.max_image_size} bytes)")
        
        # Check file extension
        file_ext = filename.lower().split('.')[-1]
        if file_ext not in self.allowed_formats:
            errors.append(f"File format '{file_ext}' not allowed. Allowed formats: {self.allowed_formats}")
        
        # Check content type
        if not content_type.startswith('image/'):
            errors.append(f"Invalid content type: {content_type}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        import re
        
        # Remove any path separators
        filename = filename.replace('/', '').replace('\\', '')
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1)
            filename = name[:250] + '.' + ext
        
        return filename
    
    def validate_json_input(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate JSON input against schema"""
        errors = []
        
        # Basic schema validation (simplified)
        for field, rules in schema.items():
            if field not in data:
                if rules.get('required', False):
                    errors.append(f"Required field '{field}' missing")
                continue
            
            value = data[field]
            field_type = rules.get('type')
            
            if field_type and not isinstance(value, field_type):
                errors.append(f"Field '{field}' must be of type {field_type.__name__}")
            
            # Length validation for strings
            if isinstance(value, str):
                min_length = rules.get('min_length')
                max_length = rules.get('max_length')
                
                if min_length and len(value) < min_length:
                    errors.append(f"Field '{field}' must be at least {min_length} characters")
                
                if max_length and len(value) > max_length:
                    errors.append(f"Field '{field}' must be at most {max_length} characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


class PermissionManager:
    """Role-based access control"""
    
    def __init__(self):
        self.permissions = {
            "admin": ["read", "write", "delete", "admin", "train", "deploy"],
            "user": ["read", "write", "train"],
            "viewer": ["read"],
            "annotator": ["read", "write"]
        }
    
    def has_permission(self, user_roles: List[str], required_permission: str) -> bool:
        """Check if user has required permission"""
        user_permissions = set()
        
        for role in user_roles:
            if role in self.permissions:
                user_permissions.update(self.permissions[role])
        
        return required_permission in user_permissions
    
    def require_permission(self, permission: str):
        """Decorator to require specific permission"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get user from kwargs (assumes user is passed as dependency)
                user = kwargs.get('current_user')
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                if not self.has_permission(user.get('roles', []), permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{permission}' required"
                    )
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


class SessionManager:
    """Manage user sessions"""
    
    def __init__(self):
        self.config = get_config()
        self.redis_client = None
        self.session_timeout = 3600  # 1 hour
    
    async def initialize(self):
        """Initialize Redis connection for sessions"""
        redis_config = self.config.get_database_config().get('redis', {})
        
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 1),  # Use different DB for sessions
            password=redis_config.get('password'),
            decode_responses=True
        )
    
    async def create_session(self, user_id: int, session_data: Dict[str, Any]) -> str:
        """Create a new session"""
        if not self.redis_client:
            await self.initialize()
        
        session_id = secrets.token_urlsafe(32)
        session_key = f"session:{session_id}"
        
        session_info = {
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            **session_data
        }
        
        await self.redis_client.setex(
            session_key,
            self.session_timeout,
            json.dumps(session_info)
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if not self.redis_client:
            return None
        
        session_key = f"session:{session_id}"
        session_data = await self.redis_client.get(session_key)
        
        if session_data:
            session_info = json.loads(session_data)
            
            # Update last activity
            session_info["last_activity"] = datetime.utcnow().isoformat()
            await self.redis_client.setex(
                session_key,
                self.session_timeout,
                json.dumps(session_info)
            )
            
            return session_info
        
        return None
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if not self.redis_client:
            return False
        
        session_key = f"session:{session_id}"
        result = await self.redis_client.delete(session_key)
        return result > 0
    
    async def cleanup_expired_sessions(self):
        """Clean up expired sessions (Redis handles this automatically)"""
        pass


class AuditLogger:
    """Security audit logging"""
    
    def __init__(self):
        self.config = get_config()
        self.audit_logger = logging.getLogger("security_audit")
    
    def log_authentication(self, username: str, success: bool, ip_address: str = None):
        """Log authentication attempts"""
        self.audit_logger.info(
            "Authentication attempt",
            extra={
                "event_type": "authentication",
                "username": username,
                "success": success,
                "ip_address": ip_address,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_api_access(self, user_id: int, endpoint: str, method: str, 
                      ip_address: str = None, user_agent: str = None):
        """Log API access"""
        self.audit_logger.info(
            "API access",
            extra={
                "event_type": "api_access",
                "user_id": user_id,
                "endpoint": endpoint,
                "method": method,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_permission_denied(self, user_id: int, endpoint: str, required_permission: str,
                            ip_address: str = None):
        """Log permission denied events"""
        self.audit_logger.warning(
            "Permission denied",
            extra={
                "event_type": "permission_denied",
                "user_id": user_id,
                "endpoint": endpoint,
                "required_permission": required_permission,
                "ip_address": ip_address,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_data_access(self, user_id: int, data_type: str, action: str, 
                       resource_id: str = None):
        """Log data access events"""
        self.audit_logger.info(
            "Data access",
            extra={
                "event_type": "data_access",
                "user_id": user_id,
                "data_type": data_type,
                "action": action,
                "resource_id": resource_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )


class SecurityScanner:
    """Basic security scanning and validation"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'\.\./',
            r'\\.\\.\\',
        ]
    
    def scan_input(self, input_data: str) -> Dict[str, Any]:
        """Scan input for suspicious patterns"""
        import re
        
        threats_found = []
        
        for pattern in self.suspicious_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                threats_found.append(pattern)
        
        return {
            "safe": len(threats_found) == 0,
            "threats": threats_found,
            "risk_level": "high" if threats_found else "low"
        }
    
    def sanitize_input(self, input_data: str) -> str:
        """Sanitize input by removing dangerous content"""
        import re
        import html
        
        # HTML escape
        sanitized = html.escape(input_data)
        
        # Remove script tags
        sanitized = re.sub(r'<script.*?>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove javascript: and vbscript: schemes
        sanitized = re.sub(r'(javascript|vbscript):', '', sanitized, flags=re.IGNORECASE)
        
        # Remove event handlers
        sanitized = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized


# Global instances
rate_limiter = RateLimiter()
api_key_manager = APIKeyManager()
permission_manager = PermissionManager()
session_manager = SessionManager()
audit_logger = AuditLogger()
security_scanner = SecurityScanner()


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance"""
    return rate_limiter


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance"""
    return api_key_manager


def get_permission_manager() -> PermissionManager:
    """Get global permission manager instance"""
    return permission_manager


def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    return session_manager


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    return audit_logger


def get_security_scanner() -> SecurityScanner:
    """Get global security scanner instance"""
    return security_scanner
