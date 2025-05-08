import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import requests
import markdown
import json
from typing import List, Tuple, Dict, Any, Optional
from http import HTTPStatus
import os

class SentimentAnalysisService:
    """Service for analyzing sentiment and summarizing product comments using AI models."""
    
    _model = None
    _tokenizer = None
    _GEMINI_API_KEY = os.getenv('GEMINI_API') 
    _GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={_GEMINI_API_KEY}"

    @classmethod
    def initialize(cls) -> None:
        """
        Initialize the sentiment analysis model and tokenizer.
        
        Raises:
            RuntimeError: If model or tokenizer initialization fails.
        """
        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(os.getenv('HUGGINGFACE_MODEL'))
            cls._model = AutoModelForSequenceClassification.from_pretrained(os.getenv('HUGGINGFACE_MODEL'))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model or tokenizer: {str(e)}")

    @classmethod
    def analyze_comments(cls, comments: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        Analyze sentiment of product comments using a pre-trained transformer model.
        
        Args:
            comments: List of comment strings to analyze.
        
        Returns:
            Tuple of (negative_comments, positive_comments, neutral_comments).
        
        Raises:
            RuntimeError: If model/tokenizer not initialized or analysis fails.
        """
        if cls._tokenizer is None or cls._model is None:
            raise RuntimeError("Model and tokenizer are not initialized")

        try:
            negative_comments, positive_comments, neutral_comments = [], [], []
            
            with torch.no_grad():
                for comment in comments:
                    inputs = cls._tokenizer(comment, return_tensors="pt", padding=True, truncation=True)
                    outputs = cls._model(**inputs)
                    sentiment = torch.softmax(outputs.logits, dim=1).argmax().item()

                    if sentiment == 0:
                        negative_comments.append(comment)
                    elif sentiment == 1:
                        positive_comments.append(comment)
                    else:
                        neutral_comments.append(comment)

            return negative_comments, positive_comments, neutral_comments

        except Exception as e:
            raise RuntimeError(f"Error analyzing comments: {str(e)}")

    @classmethod
    def summarize_comments(cls, negative_comments: List[str], positive_comments: List[str], 
                         neutral_comments: List[str]) -> str:
        """
        Summarize product comments using Gemini API and return HTML-formatted summary.
        
        Args:
            negative_comments: List of negative comment strings.
            positive_comments: List of positive comment strings.
            neutral_comments: List of neutral comment strings.
        
        Returns:
            HTML string of the summarized comments.
        
        Raises:
            requests.RequestException: If Gemini API call fails.
            ValueError: If Gemini API response is invalid.
        """
        prompt_text = (
            "Hãy tóm tắt các ý chính về sản phẩm dựa trên các bình luận sau đây, dài khoảng 100 từ.\n\n"
            "Bình luận tiêu cực:\n" + "\n".join(negative_comments or ["Không có bình luận tiêu cực."]) + "\n\n"
            "Bình luận tích cực:\n" + "\n".join(positive_comments or ["Không có bình luận tích cực."]) + "\n\n"
            "Bình luận trung lập:\n" + "\n".join(neutral_comments or ["Không có bình luận trung lập."])
        )
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text}
                    ]
                }
            ]
        }
        headers = {
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(cls._GEMINI_API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            
            if not data.get('candidates') or not data['candidates'][0].get('content'):
                raise ValueError("Invalid response from Gemini API")
            
            markdown_text = data['candidates'][0]['content']['parts'][0]['text']
            html_output = markdown.markdown(markdown_text)
            return html_output
        
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to summarize comments: {str(e)}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Invalid Gemini API response format: {str(e)}")