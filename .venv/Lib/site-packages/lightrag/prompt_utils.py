"""
Prompt Utilities for LightRAG
Tiện ích quản lý prompts cho LightRAG

This module provides utilities to automatically select appropriate prompts
based on the language setting, with optimized prompts for Vietnamese.
"""

from __future__ import annotations
from typing import Any
import re

from lightrag.prompt import PROMPTS
from lightrag.prompt_vietnamese import PROMPTS_VI


def detect_language(text: str) -> str:
    """
    Detect if text contains Vietnamese characters.
    
    Args:
        text: Input text to detect language
        
    Returns:
        'Vietnamese' if Vietnamese characters detected, otherwise 'English'
    """
    # Vietnamese diacritics and special characters
    vietnamese_pattern = re.compile(
        r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        r'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]'
    )
    
    if vietnamese_pattern.search(text):
        return 'Vietnamese'
    return 'English'


def get_prompt(
    prompt_key: str,
    language: str | None = None,
    auto_detect_text: str | None = None,
    **kwargs
) -> str | list | Any:
    """
    Get the appropriate prompt based on language setting.
    
    Args:
        prompt_key: Key of the prompt to retrieve (e.g., 'rag_response', 'keywords_extraction')
        language: Language setting ('Vietnamese', 'English', etc.). If None, will auto-detect
        auto_detect_text: Text to use for auto-detection if language is None
        **kwargs: Additional arguments for prompt formatting
        
    Returns:
        The prompt template (string or list) from appropriate prompt collection
        
    Examples:
        >>> # Explicit language selection
        >>> prompt = get_prompt('rag_response', language='Vietnamese')
        >>> 
        >>> # Auto-detection based on text
        >>> prompt = get_prompt('keywords_extraction', auto_detect_text='Kinh tế Việt Nam')
        >>> 
        >>> # Default to English if no language info
        >>> prompt = get_prompt('rag_response')
    """
    # Determine which language to use
    effective_language = language
    
    if effective_language is None and auto_detect_text:
        effective_language = detect_language(auto_detect_text)
    
    # Use Vietnamese prompts if language is Vietnamese and prompt exists
    if effective_language == 'Vietnamese' and prompt_key in PROMPTS_VI:
        prompt = PROMPTS_VI[prompt_key]
    else:
        # Fall back to default prompts
        prompt = PROMPTS.get(prompt_key)
    
    # Format prompt if kwargs provided and prompt is a string
    if kwargs and isinstance(prompt, str):
        try:
            return prompt.format(**kwargs)
        except KeyError:
            # Return unformatted if some keys are missing
            return prompt
    
    return prompt


def get_prompts_collection(language: str | None = None) -> dict[str, Any]:
    """
    Get the entire prompts collection for a given language.
    
    Args:
        language: Language setting ('Vietnamese', 'English', etc.)
        
    Returns:
        Dictionary of all prompts for the specified language
        
    Examples:
        >>> vi_prompts = get_prompts_collection('Vietnamese')
        >>> en_prompts = get_prompts_collection('English')
    """
    if language == 'Vietnamese':
        return PROMPTS_VI
    return PROMPTS


def merge_prompts_with_custom(
    custom_prompts: dict[str, Any],
    language: str | None = None
) -> dict[str, Any]:
    """
    Merge custom prompts with language-specific defaults.
    Custom prompts take precedence.
    
    Args:
        custom_prompts: Dictionary of custom prompt overrides
        language: Language setting for base prompts
        
    Returns:
        Merged dictionary with custom prompts overriding defaults
        
    Examples:
        >>> custom = {'rag_response': 'My custom prompt...'}
        >>> prompts = merge_prompts_with_custom(custom, 'Vietnamese')
    """
    base_prompts = get_prompts_collection(language).copy()
    base_prompts.update(custom_prompts)
    return base_prompts


def is_vietnamese_text(text: str, threshold: float = 0.1) -> bool:
    """
    Check if text is likely Vietnamese based on character frequency.
    
    Args:
        text: Text to check
        threshold: Minimum ratio of Vietnamese characters to total characters (0.0 to 1.0)
        
    Returns:
        True if text appears to be Vietnamese, False otherwise
        
    Examples:
        >>> is_vietnamese_text('Xin chào')
        True
        >>> is_vietnamese_text('Hello world')
        False
    """
    if not text:
        return False
    
    vietnamese_pattern = re.compile(
        r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ'
        r'ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]'
    )
    
    vietnamese_chars = len(vietnamese_pattern.findall(text))
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars == 0:
        return False
    
    ratio = vietnamese_chars / total_chars
    return ratio >= threshold


def format_vietnamese_number(number: float | int, use_comma_separator: bool = True) -> str:
    """
    Format numbers according to Vietnamese conventions.
    
    Args:
        number: Number to format
        use_comma_separator: If True, use comma for thousands separator and period for decimals
                            If False, use period for thousands and comma for decimals (Western style)
        
    Returns:
        Formatted number string
        
    Examples:
        >>> format_vietnamese_number(1234567.89)
        '1.234.567,89'
        >>> format_vietnamese_number(1234567.89, use_comma_separator=False)
        '1,234,567.89'
    """
    if isinstance(number, int):
        formatted = f"{number:,}"
        if use_comma_separator:
            # Vietnamese style: period for thousands
            formatted = formatted.replace(',', '.')
        return formatted
    
    # For floats
    parts = f"{number:,.2f}".split('.')
    integer_part = parts[0]
    decimal_part = parts[1] if len(parts) > 1 else "00"
    
    if use_comma_separator:
        # Vietnamese style: period for thousands, comma for decimals
        integer_part = integer_part.replace(',', '.')
        return f"{integer_part},{decimal_part}"
    else:
        # Western style: comma for thousands, period for decimals
        return f"{integer_part}.{decimal_part}"


# Common Vietnamese keywords for detection
VIETNAMESE_KEYWORDS = {
    'questions': ['là gì', 'như thế nào', 'tại sao', 'ở đâu', 'khi nào', 'ai', 'bao nhiêu', 'thế nào'],
    'references': ['tài liệu tham khảo', 'nguồn', 'trích dẫn'],
    'common_words': ['và', 'của', 'có', 'được', 'với', 'từ', 'trong', 'cho', 'về', 'này', 'đó'],
}


def contains_vietnamese_keywords(text: str) -> bool:
    """
    Check if text contains common Vietnamese keywords.
    
    Args:
        text: Text to check
        
    Returns:
        True if Vietnamese keywords found, False otherwise
    """
    text_lower = text.lower()
    for category in VIETNAMESE_KEYWORDS.values():
        for keyword in category:
            if keyword in text_lower:
                return True
    return False


def get_language_from_config(global_config: dict, query_text: str = "") -> str:
    """
    Determine language from global config with fallback to auto-detection.
    
    Args:
        global_config: Global configuration dictionary
        query_text: Optional query text for auto-detection
        
    Returns:
        Language string ('Vietnamese', 'English', etc.)
    """
    # First check explicit language setting
    if "addon_params" in global_config and "language" in global_config["addon_params"]:
        config_lang = global_config["addon_params"]["language"]
        # Normalize language names
        if config_lang.lower() in ['vietnamese', 'việt nam', 'tiếng việt', 'vi', 'vie', 'vn']:
            return 'Vietnamese'
        return config_lang
    
    # Fall back to auto-detection if query text provided
    if query_text:
        if is_vietnamese_text(query_text) or contains_vietnamese_keywords(query_text):
            return 'Vietnamese'
    
    # Default to English
    return 'English'
