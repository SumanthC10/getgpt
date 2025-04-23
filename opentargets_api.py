import json
import time
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Endpoints for OpenTargets APIs
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

# GraphQL query to fetch studies for a given disease (EFO ID)
STUDIES_QUERY = """
query GWASStudiesQuery($diseaseIds: [String!]!) {
  studies(diseaseIds: $diseaseIds) {
    count
    rows {
      id
      projectId
      traitFromSource
      publicationFirstAuthor
      publicationDate
      publicationJournal
      nSamples
      cohorts
      pubmedId
      ldPopulationStructure {
        ldPopulation
        relativeSampleSize
      }
    }
  }
}
"""

# Updated GraphQL query to fetch variant associations for a given study using bestGenes.
VARIANTS_QUERY = """
query StudyVariants($studyId: String!) {
  manhattan(studyId: $studyId) {
    associations {
      variant {
        id
        rsId
        chromosome
        position
        nearestCodingGene {
          id
          symbol
          __typename
        }
        nearestCodingGeneDistance
        __typename
      }
      pval
      credibleSetSize
      ldSetSize
      oddsRatio
      oddsRatioCILower
      oddsRatioCIUpper
      beta
      betaCILower
      betaCIUpper
      direction
      bestGenes {
        score
        gene {
          id
          symbol
          __typename
        }
      }
    }
  }
}
"""

# GraphQL query to fetch associated targets for a given disease
ASSOCIATED_TARGETS_QUERY = """
query AssociatedTargets($diseaseId: String!) {
  disease(efoId: $diseaseId) {
    id
    name
    associatedTargets {
      count
      rows {
        target {
          id
          approvedSymbol
        }
        score
      }
    }
  }
}
"""

def get_studies(disease_id):
    variables = {"diseaseIds": [disease_id]}
    result = execute_query(STUDIES_QUERY, variables)
    studies = result["data"]["studies"]["rows"]
    return studies

def get_variants_for_study(study_id):
    variables = {"studyId": study_id}
    result = execute_genetics_query(VARIANTS_QUERY, variables)
    associations = result["data"]["manhattan"]["associations"]
    return associations

def get_associated_targets(disease_id):
    variables = {"diseaseId": disease_id}
    result = execute_query(ASSOCIATED_TARGETS_QUERY, variables)
    disease_info = result.get("data", {}).get("disease", {})
    if not disease_info:
        raise ValueError("No disease information found")
    targets = disease_info.get("associatedTargets", {}).get("rows", [])
    associated_genes = []
    for row in targets:
        target = row.get("target", {})
        if target:
            associated_genes.append({
                "target_id": target.get("id"),
                "approvedSymbol": target.get("approvedSymbol"),
                "score": row.get("score")
            })
    return associated_genes

def test_opentargets_api():
    try:
        test_query = """
        {
          meta {
            apiVersion {
              x
              y
              z
            }
          }
        }
        """
        response = execute_query(test_query)
        if 'errors' in response:
            logger.error(f"GraphQL errors: {json.dumps(response['errors'], indent=2)}")
            return f"Error querying OpenTargets Platform API: {response['errors'][0]['message']}"
        version = response["data"]["meta"]["apiVersion"]
        platform_message = f"Successfully connected to OpenTargets Platform API. Version: {version['x']}.{version['y']}.{version['z']}"
        return platform_message
    except Exception as e:
        logger.exception("Error testing OpenTargets API")
        return f"Error testing OpenTargets API: {str(e)}"
    
def test_opentargets_genetics_api():
    """
    Tests connectivity to the OpenTargets Genetics GraphQL API by querying the dataVersion.

    Returns:
        str: Success message with version or error message.
    """
    try:
        test_query = """
        query metadataQ {
          meta {
            dataVersion {
              major
              minor
              patch
            }
          }
        }
        """
        response = execute_genetics_query(test_query)
        if 'errors' in response:
            logger.error(f"GraphQL errors: {json.dumps(response['errors'], indent=2)}")
            return f"Error querying OpenTargets Genetics API: {response['errors'][0]['message']}"
        
        version_info = response["data"]["meta"]["dataVersion"]
        version_str = f"{version_info['major']}.{version_info['minor']}.{version_info['patch']}"
        return f"Successfully connected to OpenTargets Genetics API. Data version: {version_str}"
    
    except Exception as e:
        logger.exception("Error testing OpenTargets Genetics API")
        return f"Error testing OpenTargets Genetics API: {str(e)}"