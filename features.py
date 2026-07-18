"""
features.py
============

This file extracts the 51 features that our trained model (model.pkl)
needs, for any URL the user types in.

IMPORTANT (explain this in your viva):
---------------------------------------
Our model.pkl was trained on the PhiUSIIL Phishing URL Dataset. That
dataset does NOT use only simple things like "URL length" or "number of
dots". It also uses features taken from the actual WEBPAGE (like number
of images, whether there is a password field, whether there is a favicon,
etc). So, to use this exact model, this file has to:

    1. Look at the URL text itself (fast, no internet needed).
    2. Actually DOWNLOAD the live webpage and look at its HTML
       (needs internet, and only works if the site is online).

A few features (URLSimilarityIndex, TLDLegitimateProb, URLCharProb,
CharContinuationRate, TLD encoding) were originally built using extra
private lookup tables that were not saved with model.pkl. For those,
we use a simple, clearly-labelled approximation below. This is normal
for a college mini-project and is worth mentioning in your viva.
"""

import re
import difflib
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# How long we wait for a website to respond before giving up (seconds)
REQUEST_TIMEOUT = 8

# Pretend to be a normal browser so more websites let us in
HEADERS = {"User-Agent": "Mozilla/5.0 (Phishing-Detector-College-Project)"}


class FeatureExtractionError(Exception):
    """Raised when we cannot fetch/read the given URL."""
    pass


# ------------------------------------------------------------------
# Small helper lookup tables (approximations, explained above)
# ------------------------------------------------------------------

# Rough "how trustworthy is this ending" score for common TLDs.
TLD_LEGIT_PROB = {
    "com": 0.90, "org": 0.85, "net": 0.78, "edu": 0.95, "gov": 0.96,
    "in": 0.70, "co": 0.55, "io": 0.55, "info": 0.45, "biz": 0.40,
    "xyz": 0.15, "top": 0.12, "click": 0.10, "online": 0.30,
}

# Simple fixed numeric code for common TLDs (stands in for the original
# LabelEncoder, whose exact mapping was not saved with the model).
TLD_CODE = {"com": 1, "org": 2, "net": 3, "edu": 4, "gov": 5, "in": 6,
            "co": 7, "io": 8, "info": 9, "biz": 10, "xyz": 11, "top": 12}

SOCIAL_SITES = ["facebook.com", "twitter.com", "x.com", "instagram.com",
                "linkedin.com", "youtube.com", "whatsapp.com"]


def _get_domain(url: str) -> str:
    """Return just the domain part of a URL, e.g. 'www.example.com'."""
    return urlparse(url).netloc.split(":")[0]  # remove :port if present


def _is_ip_address(domain: str) -> int:
    """Return 1 if the domain is a raw IP address (e.g. 192.168.1.1)."""
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    return 1 if re.match(ip_pattern, domain) else 0


def _get_tld(domain: str) -> str:
    """Return the last part after the final dot, e.g. 'com'."""
    parts = domain.split(".")
    return parts[-1].lower() if len(parts) > 1 else ""


def _char_continuation_rate(url: str) -> float:
    """
    Fraction of characters that are the 'same type' (letter / digit /
    special-symbol) as the character right before them.
    A high value means long runs of the same character type.
    """
    if len(url) < 2:
        return 0.0

    def char_type(c):
        if c.isalpha():
            return "letter"
        if c.isdigit():
            return "digit"
        return "special"

    same_count = sum(
        1 for i in range(1, len(url)) if char_type(url[i]) == char_type(url[i - 1])
    )
    return round(same_count / (len(url) - 1), 3)


def _url_char_prob(url: str) -> float:
    """
    Rough estimate of how 'normal' the characters in this URL are.
    We simply measure what fraction of characters are common,
    everyday URL characters (letters, digits, and . / - _ :).
    """
    if len(url) == 0:
        return 0.0
    normal_chars = sum(1 for c in url if c.isalnum() or c in "./-_:")
    return round(normal_chars / len(url), 3)


def _similarity_score(a: str, b: str) -> float:
    """Return a 0-100 similarity score between two strings."""
    if not a or not b:
        return 0.0
    return round(difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100, 2)


def _url_similarity_index(url: str, domain: str, is_ip: int) -> float:
    """
    Approximate 'how clean/normal does this URL look' score out of 100.
    We start at 100 and subtract points for classic suspicious patterns.
    (The original dataset compared URLs against a database of known
    legitimate sites, which we don't have access to here.)
    """
    score = 100.0
    if is_ip:
        score -= 40
    if "@" in url:
        score -= 20
    if url.count("-") > 2:
        score -= 10
    if domain.count(".") > 3:
        score -= 15
    if any(ch.isdigit() for ch in domain):
        score -= 10
    return max(score, 0.0)


# ------------------------------------------------------------------
# Step 1: features that only need the URL text (no internet needed)
# ------------------------------------------------------------------
def extract_url_features(url: str) -> dict:
    domain = _get_domain(url)
    tld = _get_tld(domain)
    is_ip = _is_ip_address(domain)

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    equals = url.count("=")
    qmarks = url.count("?")
    ampersands = url.count("&")
    obfuscated_chars = url.count("%")  # % is used to hide/encode characters

    # Anything that is not a letter, digit, or one of the common symbols
    common_symbols = set("=?&%")
    other_special = sum(
        1 for c in url if not c.isalnum() and c not in common_symbols
    )

    url_len = len(url) if len(url) > 0 else 1  # avoid divide-by-zero

    features = {
        "URLLength": len(url),
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "TLD": TLD_CODE.get(tld, 99),  # 99 = "unknown/other" TLD
        "URLSimilarityIndex": _url_similarity_index(url, domain, is_ip),
        "CharContinuationRate": _char_continuation_rate(url),
        "TLDLegitimateProb": TLD_LEGIT_PROB.get(tld, 0.5),
        "URLCharProb": _url_char_prob(url),
        "TLDLength": len(tld),
        "NoOfSubDomain": max(domain.count(".") - 1, 0) if not is_ip else 0,
        "HasObfuscation": 1 if obfuscated_chars > 0 else 0,
        "NoOfObfuscatedChar": obfuscated_chars,
        "ObfuscationRatio": round(obfuscated_chars / url_len, 3),
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": round(letters / url_len, 3),
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": round(digits / url_len, 3),
        "NoOfEqualsInURL": equals,
        "NoOfQMarkInURL": qmarks,
        "NoOfAmpersandInURL": ampersands,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": round(other_special / url_len, 3),
        "IsHTTPS": 1 if url.lower().startswith("https") else 0,
    }
    return features


# ------------------------------------------------------------------
# Step 2: features that need the live webpage's HTML
# ------------------------------------------------------------------
def extract_webpage_features(url: str, domain: str) -> dict:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException as exc:
        raise FeatureExtractionError(
            "Could not reach that website. Please check the URL and make "
            "sure the site is online, then try again."
        ) from exc

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    lines = html.splitlines()

    # ---- Title & matching ----
    title_tag = soup.find("title")
    title_text = title_tag.get_text().strip() if title_tag else ""

    # ---- Redirects ----
    final_domain = urlparse(response.url).netloc.split(":")[0]
    self_redirects = sum(
        1 for h in response.history
        if urlparse(h.url).netloc.split(":")[0] == domain
    )

    # ---- Forms ----
    forms = soup.find_all("form")
    has_external_form_submit = 0
    for form in forms:
        action = form.get("action", "")
        if action.startswith("http") and domain not in action:
            has_external_form_submit = 1
            break

    # ---- Links (<a> tags) ----
    links = soup.find_all("a")
    self_ref = empty_ref = external_ref = 0
    for a in links:
        href = a.get("href", "").strip()
        if href in ("", "#", "javascript:void(0)"):
            empty_ref += 1
        elif href.startswith("http") and domain not in href:
            external_ref += 1
        else:
            self_ref += 1

    page_text_lower = html.lower()

    features = {
        "LineOfCode": len(lines),
        "LargestLineLength": max((len(line) for line in lines), default=0),
        "HasTitle": 1 if title_tag else 0,
        "DomainTitleMatchScore": _similarity_score(domain, title_text),
        "URLTitleMatchScore": _similarity_score(url, title_text),
        "HasFavicon": 1 if soup.find("link", rel=lambda v: v and "icon" in v.lower()) else 0,
        "Robots": 1 if soup.find("meta", attrs={"name": "robots"}) else 0,
        "IsResponsive": 1 if soup.find("meta", attrs={"name": "viewport"}) else 0,
        "NoOfURLRedirect": len(response.history),
        "NoOfSelfRedirect": self_redirects,
        "HasDescription": 1 if soup.find("meta", attrs={"name": "description"}) else 0,
        "NoOfPopup": page_text_lower.count("window.open("),
        "NoOfiFrame": len(soup.find_all("iframe")),
        "HasExternalFormSubmit": has_external_form_submit,
        "HasSocialNet": 1 if any(s in page_text_lower for s in SOCIAL_SITES) else 0,
        "HasSubmitButton": 1 if soup.find("button", attrs={"type": "submit"}) or
        soup.find("input", attrs={"type": "submit"}) else 0,
        "HasHiddenFields": 1 if soup.find("input", attrs={"type": "hidden"}) else 0,
        "HasPasswordField": 1 if soup.find("input", attrs={"type": "password"}) else 0,
        "Bank": 1 if "bank" in page_text_lower else 0,
        "Pay": 1 if "pay" in page_text_lower else 0,
        "Crypto": 1 if "crypto" in page_text_lower else 0,
        "HasCopyrightInfo": 1 if ("©" in html or "copyright" in page_text_lower) else 0,
        "NoOfImage": len(soup.find_all("img")),
        "NoOfCSS": len(soup.find_all("link", rel="stylesheet")) + len(soup.find_all("style")),
        "NoOfJS": len(soup.find_all("script")),
        "NoOfSelfRef": self_ref,
        "NoOfEmptyRef": empty_ref,
        "NoOfExternalRef": external_ref,
    }
    return features


# ------------------------------------------------------------------
# Main function used by app.py
# ------------------------------------------------------------------
def extract_features(url: str) -> dict:
    """
    Given a URL typed by the user, return a dictionary with all 51
    feature values the model needs, in order to make a prediction.
    """
    # Add a scheme (http://) if the user forgot to type one
    if not url.lower().startswith(("http://", "https://")):
        url = "http://" + url

    domain = _get_domain(url)
    if not domain:
        raise FeatureExtractionError("That does not look like a valid URL.")

    url_features = extract_url_features(url)
    webpage_features = extract_webpage_features(url, domain)

    # Merge both dictionaries into one
    all_features = {**url_features, **webpage_features}
    return all_features
