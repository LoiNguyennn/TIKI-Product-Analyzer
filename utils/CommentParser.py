import pandas as pd
import csv
import os
import logging

def CommentParser(comment):
    """
    Parse a single comment, extracting the 'content' field and cleaning it.
    
    Args:
        comment: A dictionary containing comment data.
    
    Returns:
        str: The cleaned comment content.
    
    Raises:
        ValueError: If comment is not a dictionary or lacks 'content'.
    """
    if not isinstance(comment, dict):
        raise ValueError(f"Expected dictionary for comment, got {type(comment)}: {comment}")
    
    res = comment.get('content')
    if res is None:
        res = ''
    
    res = res.replace('\n', ' ').replace('\r', ' ')
    return res