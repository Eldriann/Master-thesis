"""
The chatbot integration component module
"""

import datetime
import logging
import json
import os
import time
import html2text
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ParseMode
from news_retrival.guardian import GuardianRetriever
from nlu.temporal.duckling_wrapper import DucklingWrapper, ExtractedResult
from nlu.topic.extractor import Extractor
from summarizer.summarizer import Summarizer


class Chatbot:
    """The class representing the chatbot that interact with telegram"""
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-few-public-methods

    help_message = "I'm a bot that search news for you! Just ask me about news on a "\
                   "subject.\n\nThe following commands are handled:\n\n/start : start the "\
                   "conversation with me\n\n/help : display this help\n\n/login [UUID]: login "\
                   "using the specified uuid\n\n/logout: terminate the logged session and remove "\
                   "the chat id from the bot memory "
    logged_users_map = {}

    def __init__(self, config) -> None:
        self.nlu = None
        self.time_parser = None
        self.retriever = None
        self.summarizer = None
        self.token = config.get('BOT_API_TOKEN')
        self.logs_dir = config.get('BOT_LOG_DIR')
        self.allowed_ids = config.get('BOT_ALLOWED_IDS').split(',')
        self.use_summarizer = (config.get('USE_SUMMARIZER').lower() in ('true', '1', 't'))
        self.use_duckling = (config.get('USE_DUCKLING').lower() in ('true', '1', 't'))
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initializing bot with token %s and log dir %s', self.token, self.logs_dir)
        self.logger.info('Allowed ids: %s', json.dumps(self.allowed_ids, indent=4))
        if self.use_summarizer:
            self.logger.info('Summarizer enabled')
        else:
            self.logger.info('Summarizer disabled')
        if self.use_duckling:
            self.logger.info('Duckling enabled')
        else:
            self.logger.info('Duckling disabled')
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        self.updater = Updater(token=self.token, use_context=True)
        self.dispatcher = self.updater.dispatcher
        self._init_handlers()

    def _is_logged(self, chat_id: str) -> bool:
        return chat_id in self.logged_users_map

    def _log_csv_event(self, user_id: str, turn_nb: int, event: str, bot_response: object) -> None:
        file_path = f'{self.logs_dir}/{user_id}.csv'
        current_time = datetime.datetime.now().isoformat()
        is_first_log = not os.path.exists(file_path)
        with open(file_path, 'a+') as file:
            if is_first_log:
                file.write('time;turn;event;bot_response\n')
            file.write(f'{current_time};{turn_nb};{event};{json.dumps(bot_response)}\n')

    def _init_handlers(self) -> None:
        self.dispatcher.add_handler(CommandHandler('start', self._start))
        self.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command),
                                                   self._on_text))
        self.dispatcher.add_handler(CommandHandler('help', self._help))
        self.dispatcher.add_handler(CommandHandler('logout', self._logout))
        self.dispatcher.add_handler(CommandHandler('login', self._login, pass_args=True))
        self.dispatcher.add_handler(MessageHandler(Filters.command, self._unknown))
        self.logger.info('Handler initialized')

    def run(self, nlu: Extractor, time_parser: DucklingWrapper,
            retriever: GuardianRetriever, summarizer: Summarizer) -> None:
        """Start the bot and run it until a termination signal is received"""
        self.nlu = nlu
        self.time_parser = time_parser
        self.retriever = retriever
        self.summarizer = summarizer
        self.logger.info('Running bot...')
        self.updater.start_polling()
        self.updater.idle()
        self.logger.info('Stopping bot...')
        self.updater.stop()
        self.logger.info('Bot stopped')

    def _start(self, update, context) -> None:
        chat_id = update.effective_chat.id
        self.logger.debug('start command for chat_id %s', str(chat_id))
        context.bot.send_message(chat_id=chat_id, text="Hi! I'm a bot delivering news, please ask "
                                                       "me about a topic after loggin in with "
                                                       "/login [UUID]! If you are lost ask for "
                                                       "/help.")

    def _help(self, update, context) -> None:
        chat_id = update.effective_chat.id
        self.logger.debug('help command for chat_id %s', str(chat_id))
        context.bot.send_message(chat_id=chat_id, text=self.help_message)

    def _login(self, update, context) -> None:
        chat_id = update.effective_chat.id
        self.logger.debug('loging command for chat_id %s', str(chat_id))
        if len(context.args) == 0:
            context.bot.send_message(chat_id=chat_id,
                                     text="/login require a uuid as argument, usage: /login [UUID]")
            return
        login_id = context.args[0]
        if not login_id in self.allowed_ids:
            context.bot.send_message(chat_id=chat_id,
                                     text=f"{login_id} is not allowed as login id.")
            return
        if self._is_logged(chat_id):
            context.bot.send_message(chat_id=chat_id,
                                     text="You are currently already logged in the session will "
                                          "be replaced.")
            self._log_csv_event(self.logged_users_map[chat_id]['uuid'],
                                self.logged_users_map[chat_id]['nb_turns'], 'SESSION_END', '')
        self.logged_users_map[chat_id] = {'uuid': login_id, 'nb_turns': 0}
        self._log_csv_event(login_id, 0, 'SESSION_START', '')
        context.bot.send_message(chat_id=chat_id,
                                 text=f"You are now logged in as {login_id}.\n\nYou can terminate "
                                      f"your session with /logout.\n\nYou can now interact with "
                                      f"the bot.")

    def _logout(self, update, context) -> None:
        chat_id = update.effective_chat.id
        self.logger.debug('logout command for chat_id %s', str(chat_id))
        if not self._is_logged(chat_id):
            context.bot.send_message(chat_id=chat_id,
                                     text="You are currently not logged in. Please login using "
                                          "/login [UUID].")
        else:
            user_data = self.logged_users_map.pop(chat_id)
            self._log_csv_event(user_data['uuid'], user_data['nb_turns'], 'SESSION_END', '')
            context.bot.send_message(chat_id=chat_id,
                                     text="I logged you out. Your chat id is no longer stored by "
                                          "me.")

    def _unknown(self, update, context) -> None:
        chat_id = update.effective_chat.id
        self.logger.debug('unknown command for chat_id %s', str(chat_id))
        context.bot.send_message(chat_id=chat_id,
                                 text="Sorry, I didn't understand that command.")

    def _on_text(self, update, context) -> None:
        # pylint: disable=line-too-long
        # We can't really shorten the 2 too long lines
        chat_id = update.effective_chat.id
        message_text = update.message.text
        self.logger.debug('on_text message for chat_id %s', str(chat_id) +
                          ', message: ' + message_text)
        if not self._is_logged(chat_id):
            context.bot.send_message(chat_id=chat_id,
                                     text="You are currently not logged in. Please login using "
                                          "/login [UUID].")
            return
        result = self.nlu.extract(message_text)
        time_result = ExtractedResult(None, None)
        if self.use_duckling:
            time_result = self.time_parser.process(result.time_info)
        self.logged_users_map[chat_id]['nb_turns'] = self.logged_users_map[chat_id]['nb_turns'] + 1
        if result.intent == 'help':
            self._log_csv_event(self.logged_users_map[chat_id]['uuid'],
                                self.logged_users_map[chat_id]['nb_turns'],
                                'NEW_TURN', {'nlu': result.to_json_encodable(), 'answer': None})
            context.bot.send_message(chat_id=chat_id, text=self.help_message)
        if result.intent == 'retrieve_news':
            if result.topic is None or len(result.topic) == 0:
                bot_response = "I didn't find any topic in your request please ask me about " \
                               "something. (ex: I want news about TOPIC last week)"
                self._log_csv_event(self.logged_users_map[chat_id]['uuid'],
                                    self.logged_users_map[chat_id]['nb_turns'],
                                    'NEW_TURN',
                                    {'nlu': result.to_json_encodable(), 'answer': bot_response})
                context.bot.send_message(chat_id=chat_id, text=bot_response)
                return
            context.bot.send_message(chat_id=chat_id,
                                     text=f"Ok! I'm searching for news about {result.topic}, "
                                          f"please wait a second.")
            if result.topic:
                guardian_articles = self.retriever.query(result.topic,
                                                         fromdate=time_result.from_date,
                                                         todate=time_result.to_date, strict=True)
                if len(guardian_articles['response']['results']) == 0:
                    guardian_articles = self.retriever.query(result.topic,
                                                             fromdate=time_result.from_date,
                                                             todate=time_result.to_date,
                                                             strict=False)
                h2t = html2text.HTML2Text()
                h2t.ignore_links = True
                h2t.ignore_emphasis = True
                h2t.ignore_images = True
                h2t.ignore_tables = True
                documents = [h2t.handle(r['fields']['body'])
                             for r in guardian_articles['response']['results'][:8]]
                if len(documents) == 0:
                    bot_response = "I sadly didn't found any article for your request..."
                    self._log_csv_event(self.logged_users_map[chat_id]['uuid'],
                                        self.logged_users_map[chat_id]['nb_turns'],
                                        'NEW_TURN',
                                        {'nlu': result.to_json_encodable(),
                                         'answer': bot_response})
                    context.bot.send_message(chat_id=chat_id, text=bot_response)
                    return
                if not self.use_summarizer:
                    sep = '\n\n'
                    bot_response = f"I found {len(documents)} articles.\n\n" \
                                   f"Here are the articles that I found:\n\n" \
                                   f"{sep.join(['['+article['webTitle']+']('+article['webUrl']+')' for article in guardian_articles['response']['results'][:8]])} "
                    self._log_csv_event(self.logged_users_map[chat_id]['uuid'],
                                        self.logged_users_map[chat_id]['nb_turns'],
                                        'NEW_TURN',
                                        {'nlu': result.to_json_encodable(),
                                         'answer': bot_response})
                    context.bot.send_message(chat_id=chat_id, parse_mode=ParseMode.MARKDOWN,
                                             disable_web_page_preview=True, text=bot_response)
                    return
                while len(documents) < 8:
                    documents.append('<PAD>')
                summarization_start_time = time.time()
                summary = self.summarizer.summarize(documents=documents, query=result.topic)
                summarization_elapsed_time = time.time() - summarization_start_time
                sep = '\n\n'
                bot_response = f"I found {len(documents)} articles, here is a summary:" \
                               f"\n\n{summary}\n\n" \
                               f"Here are the articles used to generate the summary:\n\n" \
                               f"{sep.join(['['+article['webTitle']+']('+article['webUrl']+')' for article in guardian_articles['response']['results'][:8]])} "
                self._log_csv_event(self.logged_users_map[chat_id]['uuid'],
                                    self.logged_users_map[chat_id]['nb_turns'],
                                    'NEW_TURN',
                                    {'nlu': result.to_json_encodable(),
                                     'summarization': {'elapsed_time': summarization_elapsed_time},
                                     'answer': bot_response})
                context.bot.send_message(chat_id=chat_id, parse_mode=ParseMode.MARKDOWN,
                                         disable_web_page_preview=True, text=bot_response)
