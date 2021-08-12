#!/usr/bin/env python
"""
The main module
"""

import logging
import json
import os
from dotenv import dotenv_values
from chatbot.chatbot import Chatbot
from nlu.temporal.duckling_wrapper import DucklingWrapper
from nlu.topic.extractor import Extractor
from news_retrival.guardian import GuardianRetriever
from summarizer.summarizer import Summarizer


def main():
    """main function"""
    config = {
        **dotenv_values('.env'),
        **os.environ
    }
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=int(config.get('LOG_LEVEL')))
    logging.info('Loaded .env variables:\n%s', json.dumps(config, indent=4))
    logging.debug('Debug log enabled')
    logging.info('Info log enabled')
    logging.warning('Warning log enabled')
    logging.error('Error log enabled')
    logging.critical('Critical log enabled')
    use_summarizer = (config.get('USE_SUMMARIZER').lower() in ('true', '1', 't'))
    use_duckling = (config.get('USE_DUCKLING').lower() in ('true', '1', 't'))
    bot = Chatbot(config)
    nlu = Extractor(config)
    time_parser = None
    if use_duckling:
        time_parser = DucklingWrapper(config)
    retriever = GuardianRetriever(config)
    summarizer = None
    if use_summarizer:
        summarizer = Summarizer()
    bot.run(nlu, time_parser, retriever, summarizer)


if __name__ == '__main__':
    main()
