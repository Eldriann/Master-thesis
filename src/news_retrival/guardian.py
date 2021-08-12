"""
The news retrieval component module abstracting the guardian API
"""

import logging
import datetime
from typing import List, Optional
from dataclasses import dataclass
import requests


@dataclass
class GuardianResultsResponseItemFields:
    """A class used inside GuardianResultsResponseItem for typing purposes"""
    body: str
    """The content of the article"""


@dataclass
class GuardianResultsResponseItem:
    """A class used in GuardianResultsResponse class for typing purposes"""
    # pylint: disable=invalid-name,too-many-instance-attributes
    # This is the name of the attributes the Guardian outputs
    id: str
    """The path to content"""
    sectionId: str
    """The id of the section"""
    sectionName: str
    """The name of the section"""
    webPublicationDate: datetime.datetime
    """The combined date and time of publication"""
    webTitle: str
    webUrl: str
    """The URL of the html content"""
    apiUrl: str
    """The URL of the raw content"""
    fields: GuardianResultsResponseItemFields


@dataclass
class GuardianResultsResponse:
    """A class used in GuardianResults class for typing purposes"""
    # pylint: disable=invalid-name,too-many-instance-attributes
    # This is the name of the attributes the Guardian outputs
    status: str
    """The status of the response. It refers to the state of the API. Successful
    calls will receive an "ok" even if your query did not return any results """
    userTier: str
    total: int
    """The number of results available for your search overall"""
    startIndex: int
    pageSize: int
    """The number of items returned in this call"""
    currentPage: int
    """The number of the page you are browsing"""
    pages: int
    """The total amount of pages that are in this call"""
    orderBy: str
    """The sort order used"""
    results: List[GuardianResultsResponseItem]


@dataclass
class GuardianResults:
    """Class representing the result of a query by topic to the guardian API"""
    response: GuardianResultsResponse


class GuardianRetriever:
    """A class that interact with the guardian's api"""
    # pylint: disable=too-few-public-methods

    def __init__(self, config) -> None:
        self.logger = logging.getLogger(__name__)
        self.api_key = config.get('GUARDIAN_API_KEY')
        self.logger.info('Initializing GuardianRetriever with api key %s', self.api_key)
        self.headers = {
            'api-key': self.api_key,
            'format': 'json'  # Can be json or xml
        }
        self.request_response = None
        self.base_url = 'https://content.guardianapis.com'

    def query(self, topic: str, fromdate: Optional[datetime.date],
              todate: Optional[datetime.date], strict=False) -> GuardianResults:
        """Start a query to the guardian to retrieve articles about the topic"""
        request_url = None
        if strict:
            request_url = self.base_url + '/search?q="' + topic + '"&show-fields=body'
        else:
            request_url = self.base_url + '/search?q=' + \
                          ' AND '.join(topic.split()) + '&show-fields=body'
        if fromdate is not None:
            request_url += '&from-date=' + fromdate.isoformat()
        if todate is not None:
            request_url += '&to-date=' + todate.isoformat()
        return requests.get(request_url, self.headers).json()
