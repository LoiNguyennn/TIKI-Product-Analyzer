from flask import Flask, request, jsonify, render_template
import requests
from services.TikiService import TikiService
from services.SentimentAnalysisService import SentimentAnalysisService
import asyncio
from typing import Dict, Any
from http import HTTPStatus
import logging
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

app.config['GEMINI_API'] = os.getenv('GEMINI_API')
app.config['HUGGINGFACE_MODEL'] = os.getenv('HUGGINGFACE_MODEL')

try:
    SentimentAnalysisService.initialize()
except RuntimeError as e:
    exit(1)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/tiki/analyze', methods=['POST'])
async def analyze_tiki_product():
    """
    Analyze a Tiki product based on provided URL.
    
    Expects JSON body: {"url": "https://tiki.vn/san-pham-p123456.html?spid=789"}
    Returns product information, comments, and sentiment analysis.
    """
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                "status": "fail",
                "status_code": HTTPStatus.BAD_REQUEST,
                "message": "Missing 'url' in request body"
            }), HTTPStatus.BAD_REQUEST
        
        product_url = data['url']
        
        if not product_url.startswith('https://tiki.vn/'):
            return jsonify({
                "status": "fail",
                "status_code": HTTPStatus.BAD_REQUEST,
                "message": "Invalid Tiki URL provided"
            }), HTTPStatus.BAD_REQUEST
        
        product_id, spid, seller_id = TikiService.get_product_identity(product_url)
        
        information = TikiService.get_product_info(product_id, spid, seller_id)
        
        comments = await TikiService.get_comments(product_id, spid, seller_id)
        
        if comments and isinstance(comments[0], str):
            comment_texts = comments
        else:
            comment_texts = []
            for comment in comments:
                if not isinstance(comment, dict):
                    raise ValueError(f"Expected dictionary for comment, got {type(comment)}: {comment}")
                comment_texts.append(comment)
        
        negative_comments, positive_comments, neutral_comments = SentimentAnalysisService.analyze_comments(comment_texts)
        
        summary = SentimentAnalysisService.summarize_comments(negative_comments, positive_comments, neutral_comments)
        
        return jsonify({
            "status": "success",
            "status_code": HTTPStatus.OK,
            "data": {
                "negative_comments": negative_comments,
                "positive_comments": positive_comments,
                "neutral_comments": neutral_comments,
                "information": information,
                "summary": summary
            }
        }), HTTPStatus.OK
    
    except ValueError as e:
        return jsonify({
            "status": "fail",
            "status_code": HTTPStatus.BAD_REQUEST,
            "message": str(e)
        }), HTTPStatus.BAD_REQUEST
    except (RuntimeError, requests.RequestException) as e:
        return jsonify({
            "status": "fail",
            "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "message": f"Internal server error: {str(e)}"
        }), HTTPStatus.INTERNAL_SERVER_ERROR
    except Exception as e:
        return jsonify({
            "status": "fail",
            "status_code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "message": f"Unexpected error: {str(e)}"
        }), HTTPStatus.INTERNAL_SERVER_ERROR

if __name__ == '__main__':
    app.run(debug=True)