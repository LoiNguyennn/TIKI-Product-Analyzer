import requests
import random
import aiohttp
import asyncio
from urllib.parse import urlparse, parse_qs
from typing import Tuple, List, Dict, Any, Optional
from utils.CommentParser import CommentParser

class TikiService:
    """Service to interact with Tiki.vn API for product information and reviews."""
    
    BASE_URL = "https://tiki.vn/api/v2"
    DEFAULT_HEADERS = {
        "sec-ch-ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    }
    DEFAULT_PARAMS = {
        "platform": "web",
        "version": 3
    }

    @staticmethod
    def _build_product_params(spid: str) -> Dict[str, Any]:
        """Build query parameters for product API requests."""
        params = TikiService.DEFAULT_PARAMS.copy()
        params["spid"] = spid
        return params

    @staticmethod
    def _build_review_params(product_id: str, spid: str, seller_id: str, page: Optional[int] = None) -> Dict[str, Any]:
        """Build query parameters for review API requests."""
        params = {
            "limit": 5,
            "include": "comments,contribute_info,attribute_vote_summary",
            "sort": "score|desc,id|desc,stars|all",
            "spid": spid,
            "product_id": product_id,
            "seller_id": seller_id
        }
        if page is not None:
            params["page"] = page
        return params

    @staticmethod
    def get_product_identity(product_url: str) -> Tuple[str, str, str]:
        """
        Extract product ID, seller product ID, and seller ID from a Tiki product URL.
        
        Args:
            product_url: URL of the product on Tiki.vn.
        
        Returns:
            Tuple of (product_id, spid, seller_id).
        
        Raises:
            ValueError: If the URL is invalid or missing required parameters.
            requests.RequestException: If the API request fails.
        """
        parsed_url = urlparse(product_url)
        try:
            product_id = parsed_url.path.split('-')[-1].split('.')[0][1:]
        except IndexError:
            raise ValueError("Invalid product URL format.")

        query_params = parse_qs(parsed_url.query)
        spid = query_params.get('spid', [None])[0]

        if not product_id or not spid:
            raise ValueError("Invalid product URL: missing product_id or spid.")

        try:
            response = requests.get(
                f"{TikiService.BASE_URL}/products/{product_id}",
                headers=TikiService.DEFAULT_HEADERS,
                params=TikiService._build_product_params(spid)
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch product data: {str(e)}")

        current_seller = data.get('current_seller')
        if not current_seller:
            raise ValueError("No seller information found for the product.")
        
        seller_id = current_seller.get('id')
        if not seller_id:
            raise ValueError("Seller ID not found in response.")
        
        return product_id, spid, seller_id

    @staticmethod
    async def fetch_comments_async(session: aiohttp.ClientSession, url: str, headers: Dict[str, str], 
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch comments from Tiki API asynchronously.
        
        Args:
            session: aiohttp ClientSession for making requests.
            url: API endpoint URL.
            headers: HTTP headers for the request.
            params: Query parameters for the request.
        
        Returns:
            JSON response from the API.
        
        Raises:
            aiohttp.ClientError: If the API request fails.
        """
        async with session.get(url, headers=headers, params=params) as response:
            response.raise_for_status()
            return await response.json()

    @staticmethod
    async def get_comments(product_id: str, spid: str, seller_id: str) -> List[Dict[str, Any]]:
        """
        Fetch a random sample of product comments asynchronously.
        
        Args:
            product_id: Product ID.
            spid: Seller product ID.
            seller_id: Seller ID.
        
        Returns:
            List of parsed comment dictionaries.
        
        Raises:
            aiohttp.ClientError: If the API request fails.
            ValueError: If no comments are found or paging data is invalid.
        """
        api_url = f"{TikiService.BASE_URL}/reviews"
        params = TikiService._build_review_params(product_id, spid, seller_id)

        async with aiohttp.ClientSession() as session:
            try:
                # Fetch first page to get total pages
                response = await TikiService.fetch_comments_async(session, api_url, TikiService.DEFAULT_HEADERS, params)
                paging = response.get('paging', {})
                total_pages = paging.get('last_page', 1)
                if total_pages < 1:
                    raise ValueError("Invalid paging data from API.")
                
                # Select random pages (up to 100)
                random_pages = random.sample(range(1, total_pages + 1), min(100, total_pages))
                
                # Fetch comments from random pages
                tasks = [
                    TikiService.fetch_comments_async(
                        session, api_url, TikiService.DEFAULT_HEADERS,
                        TikiService._build_review_params(product_id, spid, seller_id, page)
                    )
                    for page in random_pages
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

                comments = []
                for response in responses:
                    if isinstance(response, Exception):
                        continue
                    for comment in response.get('data', []):
                        if len(comment.get('content', '')) > 10:
                            parsed_comment = CommentParser(comment)
                            if parsed_comment:
                                comments.append(parsed_comment)
                
                return comments
            except aiohttp.ClientError as e:
                raise aiohttp.ClientError(f"Failed to fetch comments: {str(e)}")

    @staticmethod
    def get_product_info(product_id: str, spid: str, seller_id: str) -> Dict[str, Any]:
        """
        Fetch product information from Tiki API.
        
        Args:
            product_id: Product ID.
            spid: Seller product ID.
            seller_id: Seller ID (not used in API call but included for consistency).
        
        Returns:
            Dictionary containing product details.
        
        Raises:
            requests.RequestException: If the API request fails.
            ValueError: If required product information is missing.
        """
        try:
            response = requests.get(
                f"{TikiService.BASE_URL}/products/{product_id}",
                headers=TikiService.DEFAULT_HEADERS,
                params=TikiService._build_product_params(spid)
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to fetch product info: {str(e)}")

        try:
            return {
                'name': data.get('name', ''),
                'price': data.get('price', 0),
                'sold': data.get('all_time_quantity_sold', 0),
                'categories': data.get('categories', {}).get('name', ''),
                'description': data.get('description', ''),
                'images': [img.get('base_url', '') for img in data.get('images', [])],
                'rating': data.get('rating_average', 0.0)
            }
        except (KeyError, AttributeError) as e:
            raise ValueError(f"Invalid product data format: {str(e)}")