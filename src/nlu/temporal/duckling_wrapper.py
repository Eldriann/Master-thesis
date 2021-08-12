"""
The NLU time extraction component module abstracting duckling
"""

from datetime import datetime
from typing import Optional
import logging
import dataclasses
import json
import dateutil
import requests


@dataclasses.dataclass
class ExtractedResult:
    """Class keeping track of the result from duckling"""
    from_date: Optional[datetime]
    to_date: Optional[datetime]

    def to_json_encodable(self):
        """Convert a dataclass to a json encodable object"""
        return dataclasses.asdict(self)


class DucklingWrapper:
    """A class abstracting a request to a duckling service used to extract a timestamp from a
    temporal expression """
    # pylint: disable=too-few-public-methods

    def __init__(self, config) -> None:
        self.logger = logging.getLogger(__name__)
        self.base_url = config.get('DUCKLING_URL')
        self.fake_time = config.get('FAKE_DATE_NOW')
        self.logger.info('Using duckling url %s, timestamp %s', self.base_url, self.fake_time)

    def process(self, temporal_expression: str) -> ExtractedResult:
        """Process a temporal expression to return the computed timestamps"""
        data = {
                'locale': 'en_GB',
                'text': temporal_expression
        }
        if self.fake_time and len(self.fake_time):
            data = {
                'locale': 'en_GB',
                'text': temporal_expression,
                'reftime': self.fake_time
            }
        result = requests.post(self.base_url + '/parse', data=data)
        json_result = result.json()
        self.logger.debug('Temporal information parsed:')
        self.logger.debug(json.dumps(json_result, indent=4))
        from_date = None
        to_date = None
        if len(json_result) > 0:
            if json_result[0]['value']['type'] == 'value':
                from_date = dateutil.parser.parse(json_result[0]['value']['value'])
            elif json_result[0]['value']['type'] == 'interval':
                from_date = dateutil.parser.parse(json_result[0]['value']['from']['value'])
                to_date = dateutil.parser.parse(json_result[0]['value']['to']['value'])
        if to_date is None and self.fake_time and len(self.fake_time):
            to_date = datetime.fromtimestamp(int(self.fake_time) / 1000)
        return ExtractedResult(from_date=from_date, to_date=to_date)
