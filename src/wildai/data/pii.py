"""E-mail and IP-address anonymization with FineWeb's rules, for text that never went through FineWeb's PII step.

FineWeb (DataTrove's ``PIIFormatter``) replaces e-mail addresses matched by a lower-case RFC-5322-style pattern, and IPv4
addresses that are publicly routable, with fixed placeholders used in rotation. This module applies the same patterns and
placeholders; the rotation restarts for every document, so the result does not depend on processing order.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

EMAIL_PATTERN = re.compile(
    r"\b[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:(?:[a-z0-9](?:[a-z0-9-]*["
    r"a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25["
    r"0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:)])"
)
IPV4_PATTERN = re.compile(r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)")
EMAIL_PLACEHOLDERS = ("email@example.com", "firstname.lastname@example.org")
IP_PLACEHOLDERS = ("22.214.171.124", "126.96.36.199", "188.8.131.52", "184.108.40.206", "220.127.116.11", "18.104.22.168")


def _is_public_ip(candidate: str) -> bool:
    try:
        return ipaddress.ip_address(candidate).is_global
    except ValueError:
        return False


@dataclass(frozen=True)
class Anonymized:
    text: str
    emails: int
    ips: int

    @property
    def changed(self) -> bool:
        return bool(self.emails or self.ips)


def anonymize(text: str) -> Anonymized:
    """Replace e-mail addresses, then public IPv4 addresses, with FineWeb's placeholders."""

    emails = ips = 0

    def email(_match: re.Match[str]) -> str:
        nonlocal emails
        emails += 1
        return EMAIL_PLACEHOLDERS[(emails - 1) % len(EMAIL_PLACEHOLDERS)]

    def ip(match: re.Match[str]) -> str:
        nonlocal ips
        if not _is_public_ip(match.group(0)):
            return match.group(0)
        ips += 1
        return IP_PLACEHOLDERS[(ips - 1) % len(IP_PLACEHOLDERS)]

    text = EMAIL_PATTERN.sub(email, text)
    text = IPV4_PATTERN.sub(ip, text)
    return Anonymized(text, emails, ips)
