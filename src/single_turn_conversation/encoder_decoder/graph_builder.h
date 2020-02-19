#ifndef SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H
#define SINGLE_TURN_CONVERSATION_SRC_ENCODER_DECODER_GRAPH_BUILDER_H

#include <cmath>
#include <unordered_map>
#include <vector>
#include <array>
#include <set>
#include <string>
#include <memory>
#include <tuple>
#include <queue>
#include <algorithm>
#include <boost/format.hpp>
#include "N3LDG.h"
#include "tinyutf8.h"
#include "model_params.h"
#include "hyper_params.h"
#include "single_turn_conversation/print.h"
#include "single_turn_conversation/def.h"
#include "single_turn_conversation/default_config.h"
#include "single_turn_conversation/encoder_decoder/decoder_components.h"

using namespace std;
using namespace n3ldg_plus;

struct WordIdAndProbability {
    int dim;
    int word_id;
    dtype probability;

    WordIdAndProbability() = default;
    WordIdAndProbability(const WordIdAndProbability &word_id_and_probability) = default;
    WordIdAndProbability(int dimm, int wordid, dtype prob) : dim(dimm), word_id(wordid),
    probability(prob) {}
};

string getSentence(const vector<int> &word_ids_vector, const ModelParams &model_params) {
    string words;
    for (const int &w : word_ids_vector) {
        string str = model_params.lookup_table.elems.from_id(w);
        words += str;
    }
    return words;
}

class BeamSearchResult {
public:
    BeamSearchResult() {
        ngram_counts_ = {0, 0, 0};
    }
    BeamSearchResult(const BeamSearchResult &beam_search_result) = default;
    BeamSearchResult(const DecoderComponents &decoder_components,
            const vector<WordIdAndProbability> &pathh,
            dtype log_probability) : decoder_components_(decoder_components), path_(pathh),
            final_log_probability(log_probability) {
                ngram_counts_ = {0, 0, 0};
            }

    dtype finalScore() const {
//        if (path_.size() % 2 == 0) {
//            for (int n = 2; n < 10; ++n) {
//                if (path_.size() >= n * 4) {
//                    for (int i = path_.size() - n * 4 + 1; i>=0;--i) {
//                        bool ngram_hit = true;
//                        for (int j = 0; j < n; ++j) {
//                            if (path_.at(i + 2 * j).word_id != path_.at(path_.size() - n * 2 + j * 2).word_id) {
//                                ngram_hit = false;
//                                break;
//                            }
//                        }
//                        if (ngram_hit) {
//                            return -1e10;
//                        }
//                    }
//                }
//            }
//        }
        set<int> all_words, normal_words;
        for (int i = 0; i < path_.size(); ++i) {
            all_words.insert(path_.at(i).word_id);
            if (i % 2 == 1) {
                normal_words.insert(path_.at(i).word_id);
            }
        }
//        set<int> keys;
//        for (int i = 0; i < path_.size(); i +=2) {
//            keys.insert(path_.at(i).word_id);
//        }
//        return final_log_probability / path_.size();
        int len = (path_.size() % 2 == 1 ? all_words.size() : normal_words.size());
        return final_log_probability / pow(len, 1);
//        return final_log_probability / (normal_words.size() + 1e-10);
    }

    dtype finalLogProbability() const {
        return final_log_probability;
    }

    vector<WordIdAndProbability> getPath() const {
        return path_;
    }

    const DecoderComponents &decoderComponents() const {
        return decoder_components_;
    }

    void setExtraScore(dtype extra_score) {
        extra_score_ = extra_score;
    }

    dtype getExtraScore() const {
        return extra_score_;
    }

    const std::array<int, 3> &ngramCounts() const {
        return ngram_counts_;
    }

    void setNgramCounts(const std::array<int, 3> &counts) {
        ngram_counts_ = counts;
    }

private:
    DecoderComponents decoder_components_;
    vector<WordIdAndProbability> path_;
    dtype final_log_probability;
    dtype extra_score_;
    std::array<int, 3> ngram_counts_ = {};
};

string toSentenceStringWithSpace(const vector<string> &sentence) {
    string result;
    for (const string &w : sentence) {
        result += w + " ";
    }
    return result;
}

void printWordIds(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const LookupTable<Param> &lookup_table) {
    for (const WordIdAndProbability &ids : word_ids_with_probability_vector) {
        cout << lookup_table.elems.from_id(ids.word_id);
    }
    cout << endl;
}

void printWordIdsWithKeywords(const vector<WordIdAndProbability> &word_ids_with_probability_vector,
        const LookupTable<Param> &lookup_table) {
    cout << "keywords:" << endl;
    for (int i = 0; i < word_ids_with_probability_vector.size(); i += 2) {
        cout << lookup_table.elems.from_id(word_ids_with_probability_vector.at(i).word_id);
    }
    cout << endl;
    cout << "words:" << endl;
    for (int i = 1; i < word_ids_with_probability_vector.size(); i += 2) {
        int word_id = word_ids_with_probability_vector.at(i).word_id;
        cout << lookup_table.elems.from_id(word_id);
    }
    cout << endl;
//    for (int i = 1; i < word_ids_with_probability_vector.size(); i += 2) {
//        int word_id = word_ids_with_probability_vector.at(i).word_id;
//        cout << idf_table.at(lookup_table.elems.from_id(word_id)) << " ";
//    }
//    cout << endl;
}

int countNgramDuplicate(const vector<int> &ids, int n) {
    if (n >= ids.size()) {
        return 0;
    }
    vector<int> target;
    for (int i = 0; i < n; ++i) {
        target.push_back(ids.at(ids.size() - n + i));
    }

    int duplicate_count = 0;

    for (int i = 0; i < ids.size() - n; ++i) {
        bool same = true;
        for (int j = 0; j < n; ++j) {
            if (target.at(j) != ids.at(i + j)) {
                same = false;
                break;
            }
        }
        if (same) {
            ++duplicate_count;
        }
    }

    return duplicate_count;
}

void updateBeamSearchResultScore(BeamSearchResult &beam_search_result,
        const NgramPenalty& penalty) {
    vector<WordIdAndProbability> word_id_and_probability = beam_search_result.getPath();
    vector<int> ids = transferVector<int, WordIdAndProbability>(word_id_and_probability, [](
                const WordIdAndProbability &a) {return a.word_id;});
    dtype extra_score = 0.0f;
    vector<dtype> penalties = {penalty.one, penalty.two, penalty.three};
    std::array<int, 3> counts;
    for (int i = 3; i > 0; --i) {
        int duplicate_count = countNgramDuplicate(ids, i);
        counts.at(i - 1) = duplicate_count;
        extra_score -= penalties.at(i - 1) * duplicate_count;
    }
    beam_search_result.setExtraScore(beam_search_result.getExtraScore() + extra_score);
    std::array<int, 3> original_counts = beam_search_result.ngramCounts();
    std::array<int, 3> new_counts = {original_counts.at(0) + counts.at(0),
        original_counts.at(1) = counts.at(1), original_counts.at(2) + counts.at(2)};
    beam_search_result.setNgramCounts(new_counts);
}

//bool beamSearchResultCmp(const BeamSearchResult &a, const BeamSearchResult &b) {
//    return a.finalScore() != a.finalScore() ?  a.finalScore() > b.finalScore();
//}

vector<BeamSearchResult> mostProbableResults(
        const vector<DecoderComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        int current_word,
        int k,
        const ModelParams &model_params,
        const DefaultConfig &default_config,
        bool is_first,
        const vector<string> &black_list) {
    vector<Node *> nodes;
    int beam_i = 0;
    for (const DecoderComponents &decoder_components : beam) {
        auto path = last_results.at(beam_i).getPath();
        if (path.size() % 2 == 0) {
            cerr << "path is even" << endl;
            abort();
        }
        Node *node = decoder_components.wordvector_to_onehots.at(current_word);
        nodes.push_back(node);
        ++beam_i;
    }
    if (nodes.size() != last_results.size() && !last_results.empty()) {
        cerr << boost::format(
                "nodes size is not equal to last_results size, nodes size is %1% but last_results size is %2%")
            % nodes.size() % last_results.size() << endl;
        abort();
    }

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
//    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
    for (int i = 0; i < nodes.size(); ++i) {
        const Node &node = *nodes.at(i);

        BeamSearchResult beam_search_result;
        for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
            if (j == model_params.lookup_table.getElemId(::unknownkey)) {
                continue;
            }
            int stop_id = model_params.lookup_table.elems.from_string(STOP_SYMBOL);
            if (j == stop_id && !last_results.empty() &&
                    last_results.at(i).getPath().back().word_id != stop_id) {
                continue;
            }
            dtype value = node.getVal().v[j];
            dtype log_probability = log(value);
            dtype word_probability = value;
            vector<WordIdAndProbability> word_ids;
            if (!last_results.empty()) {
                log_probability += last_results.at(i).finalLogProbability();
                word_ids = last_results.at(i).getPath();
            }
            word_ids.push_back(WordIdAndProbability(node.getDim(), j, word_probability));
            beam_search_result =  BeamSearchResult(beam.at(i), word_ids, log_probability);
//            int local_size = min(k, 1 + node.getDim() / 10);
            int local_size = k;
            if (queue.size() < local_size) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                queue.pop();
                queue.push(beam_search_result);
            }
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
//        bool contain_black = false;
//        for (const string str : black_list) {
//            utf8_string utf8_str(str), utf8_sentece(sentence);
//            if (utf8_sentece.find(utf8_str) != string::npos) {
//                contain_black = true;
//                break;
//            }
//        }
//        if (contain_black) {
//            continue;
//        }
        final_results.push_back(result);
        cout << boost::format("mostProbableResults - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table);
        ++i;
    }

    return final_results;
}

vector<BeamSearchResult> mostProbableKeywords(
        DecoderComponents &beam,
        Node &keyword_onehot,
        int k,
        Graph &graph,
        ModelParams &model_params,
        const HyperParams &hyper_params,
        const DefaultConfig &default_config,
        set<int> &searched_ids,
        const vector<string> &black_list) {
    cout << "black size:" << black_list.size() << endl;
    vector<Node *> keyword_nodes, hiddens, nodes;

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;

    BeamSearchResult beam_search_result;
    for (int j = 0; j < keyword_onehot.getDim(); ++j) {
        bool should_continue = false;
        if (searched_ids.find(j) != searched_ids.end()) {
            continue;
        }
        if (j == model_params.lookup_table.getElemId(::unknownkey)) {
            continue;
        }

        for (const string &black : black_list) {
            if (black == model_params.lookup_table.elems.from_id(j)) {
                should_continue = true;
            }
        }
        if (should_continue) {
            continue;
        }
        dtype value = keyword_onehot.getVal().v[j];
        dtype log_probability = log(value);
        dtype word_probability = value;
        vector<WordIdAndProbability> word_ids;
        if (log_probability != log_probability) {
            cerr << keyword_onehot.getVal().vec() << endl;
            Json::StreamWriterBuilder builder;
            builder["commentStyle"] = "None";
            builder["indentation"] = "";
            string json_str = Json::writeString(builder, model_params.hidden_to_keyword_params.W.toJson());
            cerr << "param W:" << endl << json_str << endl;
            abort();
        }
        word_ids.push_back(WordIdAndProbability(keyword_onehot.getDim(), j, word_probability));

        BeamSearchResult local = BeamSearchResult(beam, word_ids, log_probability);
        if (queue.size() < k) {
            queue.push(local);
        } else if (queue.top().finalScore() < local.finalScore()) {
            queue.pop();
            queue.push(local);
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        if (e.finalScore() != e.finalScore()) {
            printWordIdsWithKeywords(e.getPath(), model_params.lookup_table);
            cerr << "final score nan" << endl;
            abort();
        }
        int size = e.getPath().size();
        if (size != 1) {
            cerr << boost::format("size is not 1:%1%\n") % size;
            abort();
        }
        searched_ids.insert(e.getPath().at(0).word_id);
        results.push_back(e);
        queue.pop();
    }

    vector<BeamSearchResult> final_results;
    int i = 0;
    for (const BeamSearchResult &result : results) {
        vector<int> ids = transferVector<int, WordIdAndProbability>(result.getPath(),
                [](const WordIdAndProbability &in) ->int {return in.word_id;});
        string sentence = ::getSentence(ids, model_params);
        final_results.push_back(result);
        cout << boost::format("mostProbableKeywords - i:%1% prob:%2% score:%3%") % i %
            result.finalLogProbability() % result.finalScore() << endl;
        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table);
        ++i;
    }

    return final_results;
}

struct GraphBuilder {
    DynamicLSTMBuilder left_to_right_encoder;
    Node *keyword_onehot;

    void forward(Graph &graph, const vector<string> &sentence, const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        BucketNode *hidden_bucket = new BucketNode;
        hidden_bucket->init(hyper_params.hidden_dim);
        hidden_bucket->forward(graph);
        BucketNode *word_bucket = new BucketNode;
        word_bucket->init(hyper_params.word_dim);
        word_bucket->forward(graph);

        for (int i = 0; i < sentence.size(); ++i) {
            LookupNode<Param>* input_lookup(new LookupNode<Param>);
            input_lookup->init(hyper_params.word_dim);
            input_lookup->setParam(model_params.lookup_table);
            input_lookup->forward(graph, sentence.at(i));

            DropoutNode* dropout_node(new DropoutNode(hyper_params.dropout, is_training));
            dropout_node->init(hyper_params.word_dim);
            dropout_node->forward(graph, *input_lookup);

            left_to_right_encoder.forward(graph, model_params.left_to_right_encoder_params,
                    *dropout_node, *hidden_bucket, *hidden_bucket, hyper_params.dropout,
                    is_training);
        }

        vector<Node *> keyword_hiddens;
        for (int i = 0; i < sentence.size(); ++i) {
            Node *keyword_hidden = n3ldg_plus::linear(graph,
                    model_params.encoder_to_keyword_hidden_params,
                    *left_to_right_encoder._hiddens.at(i));
            keyword_hiddens.push_back(keyword_hidden);
        }
        keyword_onehot = n3ldg_plus::maxPool(graph, keyword_hiddens);
        keyword_onehot = n3ldg_plus::linear(graph, model_params.hidden_to_keyword_params,
                *keyword_onehot);
        keyword_onehot = n3ldg_plus::linearWordVector(graph, model_params.lookup_table.nVSize,
                model_params.lookup_table.E, *keyword_onehot);
        keyword_onehot = n3ldg_plus::softmax(graph, *keyword_onehot);
    }

    void forwardDecoder(Graph &graph, DecoderComponents &decoder_components,
            const std::vector<std::string> &answer,
            const std::string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        Node *keyword_lookup = n3ldg_plus::embedding(graph, model_params.lookup_table, keyword);
        keyword_lookup = n3ldg_plus::dropout(graph, *keyword_lookup, hyper_params.dropout,
                is_training);
        decoder_components.keyword_embedding = keyword_lookup;

        int keyword_bound = model_params.lookup_table.nVSize;
        int normal_bound = model_params.lookup_table.nVSize;

        for (int i = 0; i < answer.size(); ++i) {
            forwardDecoderByOneStep(graph, decoder_components, i,
                    i == 0 ? nullptr : &answer.at(i - 1), keyword, i == 0, hyper_params,
                    model_params, is_training, keyword_bound, normal_bound);
        }
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const std::string &keyword,
            bool should_predict_keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training,
            int keyword_word_id_upper_open_bound,
            int normal_word_id_upper_open_bound) {
        if (keyword_word_id_upper_open_bound > model_params.lookup_table.nVSize) {
            cerr << boost::format("word_id_upper_open_bound:%1% vsize:%2%") %
                keyword_word_id_upper_open_bound % model_params.lookup_table.nVSize << endl;
            abort();
        }

        if (normal_word_id_upper_open_bound > keyword_word_id_upper_open_bound) {
            cerr << boost::format("normal:%1% keyword:%2%") %
                normal_word_id_upper_open_bound % keyword_word_id_upper_open_bound << endl;
            abort();
        }

        Node *last_input;
        if (i > 0) {
            LookupNode<Param>* before_dropout(new LookupNode<Param>);
            before_dropout->init(hyper_params.word_dim);
            before_dropout->setParam(model_params.lookup_table);
            before_dropout->forward(graph, *answer);

            DropoutNode* decoder_lookup(new DropoutNode(hyper_params.dropout, is_training));
            decoder_lookup->init(hyper_params.word_dim);
            decoder_lookup->forward(graph, *before_dropout);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            if (decoder_components.decoder_lookups.size() != i) {
                cerr << boost::format("decoder_lookups size:%1% i:%2%") %
                    decoder_components.decoder_lookups.size() % i << endl;
                abort();
            }
            last_input = decoder_components.decoder_lookups.back();
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
        }

        decoder_components.forward(graph, hyper_params, model_params, *last_input,
                left_to_right_encoder._hiddens, is_training);

        Node* decoder_to_wordvector = decoder_components.decoderToWordVector(graph, hyper_params,
                model_params, left_to_right_encoder._hiddens, i);
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        LinearWordVectorNode *wordvector_to_onehot(new LinearWordVectorNode);
        wordvector_to_onehot->init(normal_word_id_upper_open_bound);
        wordvector_to_onehot->setParam(model_params.lookup_table.E);
        wordvector_to_onehot->forward(graph, *decoder_to_wordvector);

        Node *softmax = n3ldg_plus::softmax(graph, *wordvector_to_onehot);

        decoder_components.wordvector_to_onehots.push_back(softmax);
    }

    void forwardDecoderHiddenByOneStep(Graph &graph, DecoderComponents &decoder_components,
            int i,
            const std::string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        Node *last_input;
        if (i > 0) {
            LookupNode<Param>* before_dropout(new LookupNode<Param>);
            before_dropout->init(hyper_params.word_dim);
            before_dropout->setParam(model_params.lookup_table);
            before_dropout->forward(graph, *answer);

            DropoutNode* decoder_lookup(new DropoutNode(hyper_params.dropout, false));
            decoder_lookup->init(hyper_params.word_dim);
            decoder_lookup->forward(graph, *before_dropout);
            decoder_components.decoder_lookups.push_back(decoder_lookup);
            if (decoder_components.decoder_lookups.size() != i) {
                cerr << boost::format("decoder_lookups size:%1% i:%2%") %
                    decoder_components.decoder_lookups.size() % i << endl;
                abort();
            }
            last_input = decoder_components.decoder_lookups.back();
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
        }

        decoder_components.forward(graph, hyper_params, model_params, *last_input,
                left_to_right_encoder._hiddens, false);
    }

    void forwardKeywordEmbedding(Graph &graph, DecoderComponents &decoder_components,
            const string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        LookupNode<Param> *keyword_lookup = new LookupNode<Param>;
        keyword_lookup->init(hyper_params.word_dim);
        keyword_lookup->setParam(model_params.lookup_table);
        keyword_lookup->forward(graph, keyword);

        Node *dropout_keyword = n3ldg_plus::dropout(graph, *keyword_lookup, hyper_params.dropout,
                false);
        decoder_components.keyword_embedding = dropout_keyword;
    }

    void forwardOnehotByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node*> &encoder_hiddens) {
        Node* result =  decoder_components.decoderToWordVector(graph, hyper_params, model_params,
                encoder_hiddens, i);

        LinearWordVectorNode *one_hot_node = new LinearWordVectorNode;
        one_hot_node->init(model_params.lookup_table.nVSize);
        one_hot_node->setParam(model_params.lookup_table.E);
        one_hot_node->forward(graph, *result);
        Node *softmax = n3ldg_plus::softmax(graph, *one_hot_node);
        decoder_components.wordvector_to_onehots.push_back(softmax);
    }

    pair<vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            Node &keyword_onehot,
            const vector<DecoderComponents> &decoder_components_beam,
            int k,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            const DefaultConfig &default_config,
            const vector<string> &black_list) {
        vector<pair<vector<WordIdAndProbability>, dtype>> word_ids_result;
        vector<BeamSearchResult> most_probable_results;
        vector<string> last_answers, last_keywords;
        set<int> searched_ids;
        bool succeeded = false;

        auto beam = decoder_components_beam;
        cout << boost::format("beam size:%1%\n") % beam.size();

        for (int iter = 0; ; ++iter) {
            cout << "iter:" << iter << endl;
            most_probable_results.clear();
            auto beam = decoder_components_beam;
            int closured_count = word_ids_result.size();
            if (closured_count >= default_config.result_count_factor * k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                if (word_ids_result.size() >= default_config.result_count_factor * k ||
                        i > default_config.cut_length) {
                    break;
                }

                if (i == 0) {
                    most_probable_results = mostProbableKeywords(beam.front(), keyword_onehot, k,
                            graph, model_params, hyper_params,
                            default_config, searched_ids, black_list);
                    for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                        DecoderComponents &decoder_components = beam.at(beam_i);
                        auto path = most_probable_results.at(beam_i).getPath();
                        if (path.size() != 1) {
                            cerr << "path size:" << path.size() << endl;
                            abort();
                        }
                        int keyword_id = path.back().word_id;
                        string keyword = model_params.lookup_table.elems.from_id(keyword_id);
                        forwardKeywordEmbedding(graph, decoder_components, keyword, hyper_params,
                                model_params);
                    }
                }
                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderComponents &decoder_components = beam.at(beam_i);
                    forwardDecoderHiddenByOneStep(graph, decoder_components, i,
                            i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                            model_params);
                }
                cout << "forwardDecoderResultByOneStep:" << endl;
                graph.compute();
                beam.clear();
                vector<BeamSearchResult> search_results;

                for (int j = 0; j < most_probable_results.size(); ++j) {
                    BeamSearchResult &beam_search_result = most_probable_results.at(j);
                    const vector<WordIdAndProbability> &word_ids = beam_search_result.getPath();
                    int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                    const string &word = model_params.lookup_table.elems.from_id(last_word_id);
                    search_results.push_back(beam_search_result);
                    last_keywords.push_back(word);
                    beam.push_back(beam_search_result.decoderComponents());
                }
                most_probable_results = search_results;

                int beam_i = 0;
                for (auto &decoder_components : beam) {
                    forwardOnehotByOneStep(graph, decoder_components, i, hyper_params,
                            model_params, left_to_right_encoder._hiddens);
                    ++beam_i;
                }
                last_keywords.clear();
                cout << "forwardDecoderKeywordByOneStep:" << endl;
                graph.compute();

                last_answers.clear();
                most_probable_results = mostProbableResults(beam, most_probable_results, i,
                        k, model_params, default_config, i == 0, black_list);
                cout << boost::format("most_probable_results size:%1%") %
                    most_probable_results.size() << endl;
                beam.clear();
                vector<BeamSearchResult> stop_removed_results;
                int j = 0;
                for (BeamSearchResult &beam_search_result : most_probable_results) {
                    const vector<WordIdAndProbability> &word_ids = beam_search_result.getPath();

                    int last_word_id = word_ids.at(word_ids.size() - 1).word_id;
                    const string &word = model_params.lookup_table.elems.from_id(
                            last_word_id);
                    if (word == STOP_SYMBOL) {
                        word_ids_result.push_back(make_pair(word_ids,
                                    beam_search_result.finalScore()));
                        succeeded = word == STOP_SYMBOL;
                    } else {
                        stop_removed_results.push_back(beam_search_result);
                        last_answers.push_back(word);
                        beam.push_back(beam_search_result.decoderComponents());
                    }
                    ++j;
                }
                most_probable_results = stop_removed_results;

                if (beam.empty()) {
                    cout << boost::format("break for beam empty\n");
                    break;
                }
            }
        }

        if (word_ids_result.size() < default_config.result_count_factor * k) {
            cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << endl;
            abort();
        }

        cout << endl<< "final search results:" << endl;
        for (const auto &pair : word_ids_result) {
            const vector<WordIdAndProbability> ids = pair.first;
            cout << boost::format("beam result:%1%") % exp(pair.second) << endl;
            printWordIdsWithKeywords(ids, model_params.lookup_table);
        }

        auto compair = [](const pair<vector<WordIdAndProbability>, dtype> &a,
                const pair<vector<WordIdAndProbability>, dtype> &b) {
            return a.second < b.second;
        };
        auto max = max_element(word_ids_result.begin(), word_ids_result.end(), compair);

        return make_pair(max->first, exp(max->second));
    }
};

#endif
