"""
The summarization component module abstracting the QmdsCnnIr library
"""
# pylint: disable=import-error,wrong-import-position
from __future__ import division

from typing import List
import logging
from collections import namedtuple
import string
import sys
import sentencepiece
import torch
sys.path.insert(0, 'libs/QmdsCnnIr/src')
from abstractive.model_builder import Summarizer as QmdsCnnlrSummarizer
from abstractive.neural import tile
from abstractive.beam_search import BeamSearch
from abstractive.beam import GNMTGlobalScorer


class Summarizer:
    """A class that summarize the text passed as arguments"""

    model_flags = [
        'emb_size', 'enc_hidden_size', 'dec_hidden_size', 'enc_layers', 'dec_layers',
        'block_size', 'heads', 'ff_size', 'hier', 'inter_layers',
        'inter_heads', 'block_size', 'attn_threshold']
    article_token_truncate = 200
    summary_min_len = 35
    summary_max_len = 500
    vocabulary_path = 'models/spm.model'
    model_path = 'models/model_step_0.251829_405000.pt'
    args_dict = {
        'log_file': '',
        'mode': 'test',
        'visible_gpus': '-1',
        'data_path': 'data/qmdscnn/pytorch_QMDS_adv/CNNDM',
        'model_path': model_path,
        'vocab_path': vocabulary_path,
        'train_from': '',
        'trunc_src_ntoken': article_token_truncate,
        'trunc_tgt_ntoken': 120,
        'emb_size': 256,
        'query_layers': 1,
        'enc_layers': 8,
        'dec_layers': 1,
        'enc_dropout': 0.1,
        'dec_dropout': 0.1,
        'enc_hidden_size': 256,
        'dec_hidden_size': 256,
        'heads': 8,
        'ff_size': 1024,
        'hier': True,
        'model_type': 'he',
        'query': False,
        'fine_tune': False,
        'batch_size': 8000,
        'valid_batch_size': 100000,
        'optim': 'adam',
        'lr': 1,
        'max_grad_norm': 0,
        'seed': 666,
        'train_steps': 500000,
        'save_checkpoint_steps': 5000,
        'max_num_checkpoints': 3,
        'report_every': 100,
        'accum_count': 1,
        'world_size': 1,
        'gpu_ranks': '0',
        'share_embeddings': True,
        'share_decoder_embeddings': True,
        'max_generator_batches': 32,
        'test_all': False,
        'test_from': model_path,
        'result_path': 'results/model-CNNDM-he/outputs',
        'alpha': 0.4,
        'length_penalty': 'wu',
        'block_ngram_repeat': 3,
        'coverage_penalty': 'summary',
        'cov_beta': 5,
        'attn_threshold': 0,
        'stepwise_penalty': True,
        'beam_size': 5,
        'n_best': 1,
        'max_length': summary_max_len,
        'min_length': summary_min_len,
        'report_rouge': True,
        'save_criteria': 'rouge_l_f_score',
        'rouge_path': None,
        'dataset': 'CNNDM',
        'max_samples': 100000,
        'inter_layers': '6,7',
        'inter_heads': 8,
        'trunc_src_nblock': 8,
        'beta1': 0.9,
        'beta2': 0.998,
        'warmup_steps': 8000,
        'decay_method': 'noam',
        'label_smoothing': 0.1,
        'lambda_dis': 0.0,
        'lambda_cov': 0.0
    }

    def __init__(self) -> None:
        # pylint: disable=no-member
        # pylint doesn't like cython libraries
        self.logger = logging.getLogger(__name__)
        self.args = namedtuple("Namespace", self.args_dict.keys())(*self.args_dict.values())
        device = "cpu" if self.args.visible_gpus == '-1' else "cuda"
        self.logger.info('Loading checkpoint from %s', self.args.test_from)
        self.logger.info('Device is %s', device)
        checkpoint = torch.load(self.args.test_from, map_location=torch.device(device))
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if k in self.model_flags:
                self.args = self.args._replace(**{k: opt[k]})
        self.logger.info('args are:')
        self.logger.info(self.args)

        # Load tokenizer
        self.logger.info('Loading tokenizer...')
        spm = sentencepiece.SentencePieceProcessor()
        spm.Load(self.args.vocab_path)
        word_padding_id = spm.PieceToId('<PAD>')
        self.symbols = {'BOS': spm.PieceToId('<S>'),
                        'EOS': spm.PieceToId('</S>'),
                        'PAD': word_padding_id,
                        'EOT': spm.PieceToId('<T>'),
                        'EOP': spm.PieceToId('<P>'),
                        'EOQ': spm.PieceToId('<Q>')}
        vocab_size = len(spm)
        self.vocab = spm
        self.logger.info('Loaded tokenizer')

        # Load model
        self.logger.info('Loading model...')
        self.model = QmdsCnnlrSummarizer(self.args, word_padding_id, vocab_size, device, checkpoint)
        self.model.eval()
        self.logger.info('Loaded model')
        self.logger.info('Summarizer initialized')

    def summarize(self, documents: List[str], query: str) -> str:
        """Summarize the given list of documents using the query"""
        # pylint: disable=too-many-locals,no-member
        # pylint doesn't like cython libraries
        # clean data and translate
        data_src = [self._clean_data(article) for article in documents]
        data_query = self._clean_data(query)
        data_src[0] = data_query + '<T> ' + data_src[0]  # Concatenate query with first article

        # Tokenize data src and truncate to 'trunc_src_ntoken'
        # keep only 'trunc_src_nblock' blocks also convert list to tensor
        src_list = [self.vocab.encode_as_ids(article_str)[:self.args.trunc_src_ntoken]
                    for article_str in data_src][:self.args.trunc_src_nblock]
        # pad data to convert to tensor
        pad_width = max([len(d) for d in src_list])
        pad_height = len(src_list)
        pad_id = self.symbols['PAD']
        src_list_padded = [d + [pad_id] * (pad_width - len(d)) for d in src_list]
        src_list_padded = src_list_padded + [[pad_id] * pad_width] * (pad_height - len(src_list))
        src = torch.tensor([src_list_padded])

        # Forward pass model
        with torch.no_grad():  # We don't need the gradients
            # Encoder forward
            batch_size, n_blocks, num_tokens = src.shape
            src_features, mask_hier = self.model.encoder(src)
            dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
            src_features = src_features.view(n_blocks, num_tokens, batch_size, -1).contiguous()
            device = src_features.device
            dec_states.map_batch_fn(lambda state, dim: tile(state, self.args.beam_size, dim=dim))
            src_features = tile(src_features, self.args.beam_size, dim=2)
            mask = tile(mask_hier, self.args.beam_size, dim=0)

            # Generate output with decoder
            beam = BeamSearch(
                self.args.beam_size,
                n_best=self.args.n_best,
                batch_size=batch_size,
                global_scorer=GNMTGlobalScorer(alpha=self.args.alpha,
                                               beta=self.args.cov_beta,
                                               length_penalty='wu',
                                               coverage_penalty=self.args.coverage_penalty),
                pad=self.symbols['PAD'],
                eos=self.symbols['EOS'],
                bos=self.symbols['BOS'],
                min_length=self.args.min_length,
                ratio=0.,
                max_length=self.args.max_length,
                mb_device=device,
                return_attention=False,
                stepwise_penalty=self.args.stepwise_penalty,
                block_ngram_repeat=self.args.block_ngram_repeat,
                exclusion_tokens=set([]),
                memory_lengths=mask)

            predictions = None
            for step in range(self.args.max_length):
                decoder_input = beam.current_predictions.view(1, -1)
                dec_out, dec_states, attn = self.model.decoder(decoder_input, src_features,
                                                               dec_states, memory_masks=mask,
                                                               step=step)

                # Generator forward.
                log_probs = self.model.generator.forward(dec_out.squeeze(0))
                attn = attn.transpose(1,2).transpose(0,1)
                attn = attn.max(2)[0]
                beam.advance(log_probs, attn)
                any_beam_is_finished = beam.is_finished.any()
                if any_beam_is_finished:
                    beam.update_finished()
                    if beam.done:
                        break

                select_indices = beam.current_origin
                if any_beam_is_finished:
                    src_features = src_features.index_select(2, select_indices)
                    mask = mask.index_select(0, select_indices)
                # pylint: disable=cell-var-from-loop
                # it is normal here
                dec_states.map_batch_fn(lambda state, dim: state.index_select(dim, select_indices))
                predictions = beam.predictions

            # Decode generated summary
            pred = sum([self._build_target_tokens(predictions[0][n], self.vocab, self.symbols)
                        for n in range(self.args.n_best)], [])
            pred_str = ' '.join(pred).replace('<Q>', ' ').replace('</S>', ' ')\
                .replace(r' +', ' ').replace('<unk>', 'UNK').strip()
            pred_str = ' '.join(pred_str.split())
            result_summary = self.clean_pred(pred_str)
            self.logger.debug('Produced output:')
            self.logger.debug(result_summary)
            return result_summary

    @staticmethod
    def _build_target_tokens(prediction, vocab, symbols):
        tokens = []
        for tok in prediction:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == symbols['EOS']:
                tokens = tokens[:-1]
                break
        tokens = [t for t in tokens if t < len(vocab)]
        tokens = vocab.DecodeIds(tokens).split(' ')
        return tokens

    @staticmethod
    def _clean_data(data: str):
        if data == '<PAD>':
            return data
        mapping_table = [
            ("'", ' '),
            ('<', ''),
            ('>', ''),
            ('\n', ' '),
            ('%', ' percent'),
            ('#', ''),
            ('*', ''),
            ('@', ' at '),
            ('(', ''),
            (')', ''),
            (';', ','),
            ('[', ''),
            (']', ''),
            ('!', '.')
        ]
        data = data.lower()
        data = ''.join(filter(lambda x: x in set(string.printable), data))
        for elem in mapping_table:
            key, replacement = elem
            data = data.replace(key, replacement)
        return data.strip()

    @staticmethod
    def clean_pred(pred_str: str):
        """Clean and capitalize a predicted string so that it can be used in markdown responses"""
        cleaned_str = ''
        is_new_sentence = True
        for idx, char in enumerate(pred_str):
            if is_new_sentence and char.isalpha():
                cleaned_str += char.upper()
                is_new_sentence = False
                continue
            if char == ' ' and len(pred_str) != idx and \
                    (pred_str[idx + 1] == '.' or pred_str[idx + 1] == ','):
                continue
            if char == '.':
                is_new_sentence = True
            if char not in ("'", '`', '*', '_'):
                cleaned_str += char
        return cleaned_str
