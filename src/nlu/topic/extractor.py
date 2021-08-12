"""
The NLU topic extraction component module abstracting rasa
"""

import json
import logging
import dataclasses
from dataclasses import dataclass
from rasa.nlu.model import Interpreter


@dataclass
class ExtractedResult:
    """Class keeping track of the extracted result from the NLU"""
    intent: str
    topic: str
    time_info: str

    def to_json_encodable(self):
        """Convert a dataclass to a json encodable object"""
        return dataclasses.asdict(self)


class Extractor:
    """A class that will extract the topic out of a sentence"""
    # pylint: disable=too-few-public-methods

    def __init__(self, config) -> None:
        self.logger = logging.getLogger(__name__)
        self.model_path = config.get('MODEL_PATH')
        self.logger.info('Using model %s', self.model_path)
        self.interpreter = Interpreter.load(self.model_path)
        self.logger.info('Model %s ready', self.model_path)

    @staticmethod
    def _compute_topic(interpretation) -> str:
        # A common error in the model is to extract "the" as a temporal info or topic we simply
        # ignore it. If the model extracted multiple topic we append them to create one topic
        topic = ''
        entities = interpretation['entities']
        for entity in entities:
            if entity['entity'] == 'topic':
                if entity['value'] != 'the':
                    topic = ' '.join([topic, entity['value']])
        return topic.strip()

    @staticmethod
    def _compute_time_info(interpretation) -> str:
        time_info = ''
        entities = interpretation['entities']
        for entity in entities:
            if entity['entity'] == 'temporal':
                if entity['value'] != 'the':
                    time_info = ' '.join([time_info, entity['value']])
        return time_info.strip()

    def extract(self, query: str) -> ExtractedResult:
        """Process a given user query"""
        interpretation = self.interpreter.parse(query)
        self.logger.debug(json.dumps(interpretation, indent=4))
        intent = interpretation['intent']['name']
        topic = self._compute_topic(interpretation)
        time_info = self._compute_time_info(interpretation)
        return ExtractedResult(intent=intent, topic=topic, time_info=time_info)
