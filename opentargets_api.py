import json
import time
import logging
import requests

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
GENETICS_GRAPHQL_URL = "https://api.genetics.opentargets.org/graphql"

def execute_query(query, variables=None, max_retries=3, initial_delay=1):
    payload = {"query": query, "variables": variables} if variables else {"query": query}
    logger.debug(f"Sending query to OpenTargets API: {json.dumps(payload, indent=2)}")
    for attempt in range(max_retries):
        try:
            response = requests.post(GRAPHQL_URL, json=payload)
            logger.debug(f"Full API Response: {response.text}")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400) - Full response: {response.text}")
                raise ValueError(f"Bad Request (400) - API Response: {response.text}")
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))
    raise Exception("Max retries reached")

def execute_genetics_query(query, variables=None, max_retries=3, initial_delay=1):
    payload = {"query": query, "variables": variables} if variables else {"query": query}
    logger.debug(f"Sending query to OpenTargets Genetics API: {json.dumps(payload, indent=2)}")
    for attempt in range(max_retries):
        try:
            response = requests.post(GENETICS_GRAPHQL_URL, json=payload)
            logger.debug(f"Full Genetics API Response: {response.text}")
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                logger.error(f"Bad Request (400) - Full response: {response.text}")
                raise ValueError(f"Bad Request (400) - Genetics API Response: {response.text}")
            else:
                response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                raise
            time.sleep(initial_delay * (2 ** attempt))
    raise Exception("Max retries reached")
