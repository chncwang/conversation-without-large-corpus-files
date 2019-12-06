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
//        return final_log_probability / (path_.size() % 2 == 1 ? all_words.size() :
//                normal_words.size());
        return final_log_probability / (normal_words.size() + 1e-10);
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
        const LookupTable<Param> &lookup_table,
        const unordered_map<string, float> &idf_table) {
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
        const vector<string> &black_list,
        const unordered_map<string, float> &idf_table) {
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
        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table, idf_table);
        ++i;
    }

    return final_results;
}

vector<BeamSearchResult> mostProbableKeywords(
        vector<DecoderComponents> &beam,
        const vector<BeamSearchResult> &last_results,
        const unordered_map<string ,float> word_idf_table,
        int word_pos,
        int k,
        Graph &graph,
        ModelParams &model_params,
        const HyperParams &hyper_params,
        const DefaultConfig &default_config,
        bool is_first,
        set<int> &searched_ids,
        const vector<string> &black_list) {
    cout << "black size:" << black_list.size() << endl;
    vector<Node *> keyword_nodes, hiddens, nodes;
    for (int ii = 0; ii < beam.size(); ++ii) {
        bool should_predict_keyword;
        if (last_results.empty()) {
            should_predict_keyword = true;
        } else {
            vector<WordIdAndProbability> path = last_results.at(ii).getPath();
            int size = path.size();
            should_predict_keyword = path.at(size - 2).word_id == path.at(size - 1).word_id;
        }
        Node *node, *keyword_node, *hidden;
        hidden = beam.at(ii).decoder._hiddens.at(word_pos);
        if (should_predict_keyword) {
            DecoderComponents &components = beam.at(ii);

            ConcatNode *concat_node = new ConcatNode();
            concat_node->init(hyper_params.hidden_dim);
            if (components.decoder_lookups.size() != word_pos) {
                cerr << boost::format("size:%1% word_pos:%2%") % components.decoder_lookups.size()
                    % word_pos << endl;
                abort();
            }

            ConcatNode *context_concated = new ConcatNode;
            context_concated->init(2 * hyper_params.hidden_dim);
            context_concated->forward(graph, {components.decoder._hiddens.at(word_pos),
                    components.contexts.at(word_pos)});

            Node *keyword = n3ldg_plus::linear(graph, model_params.hidden_to_keyword_params,
                    *context_concated);
            keyword_node = keyword;

            int last_keyword_id;
            if (last_results.empty()) {
                last_keyword_id = model_params.lookup_table.nVSize - 1;
            } else {
                vector<WordIdAndProbability> path = last_results.at(ii).getPath();
                last_keyword_id = path.at(path.size() - 2).word_id;
            }

            LinearWordVectorNode *keyword_vector_to_onehot = new LinearWordVectorNode;
            keyword_vector_to_onehot->init(last_keyword_id + 1);
            keyword_vector_to_onehot->setParam(model_params.lookup_table.E);
            keyword_vector_to_onehot->forward(graph, *keyword);

            Node *softmax = n3ldg_plus::softmax(graph, *keyword_vector_to_onehot);

            components.keyword_vector_to_onehots.push_back(softmax);
            node = softmax;
        } else {
            node = nullptr;
            keyword_node = nullptr;
        }
        nodes.push_back(node);
        keyword_nodes.push_back(keyword_node);
        hiddens.push_back(hidden);
    }
    graph.compute();

    auto cmp = [](const BeamSearchResult &a, const BeamSearchResult &b) {
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
//    for (int i = 0; i < nodes.size(); ++i) {
        const Node *node_ptr = nodes.at(i);
        if (node_ptr == nullptr) {
            vector<WordIdAndProbability> new_id_and_probs = last_results.at(i).getPath();
            WordIdAndProbability &last_keyword = new_id_and_probs.at(new_id_and_probs.size() - 2);
            WordIdAndProbability &last_norm = new_id_and_probs.at(new_id_and_probs.size() - 1);
            WordIdAndProbability w = {last_keyword.dim, last_keyword.word_id,
                last_norm.probability};
            new_id_and_probs.push_back(w);
            BeamSearchResult beam_search_result(beam.at(i), new_id_and_probs,
                    last_results.at(i).finalLogProbability());
            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                queue.pop();
                queue.push(beam_search_result);
            }
        } else {
            const Node &node = *nodes.at(i);

            BeamSearchResult beam_search_result;
            for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
                bool should_continue = false;
                if (is_first) {
                    if (searched_ids.find(j) != searched_ids.end()) {
                        continue;
                    }
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
                const string &word = model_params.lookup_table.elems.from_id(j);
                if (word_pos == 0 && word_idf_table.at(word) <= default_config.keyword_bound) {
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
                if (log_probability != log_probability) {
                    cerr << node.getVal().vec() << endl;
                    cerr << "keyword node:" << endl << keyword_nodes.at(i)->getVal().vec() << endl;
                    cerr << "hidden node:" << endl << hiddens.at(i)->getVal().vec() << endl;
                    Json::StreamWriterBuilder builder;
                    builder["commentStyle"] = "None";
                    builder["indentation"] = "";
                    string json_str = Json::writeString(builder, model_params.hidden_to_keyword_params.W.toJson());
                    cerr << "param W:" << endl << json_str << endl;
                    abort();
                }
                word_ids.push_back(WordIdAndProbability(node.getDim(), j, word_probability));

                BeamSearchResult local = BeamSearchResult(beam.at(i), word_ids, log_probability);
//                if (local_queue.size() < min(k, node.getDim() / 10 + 1)) {
                if (queue.size() < k) {
                    queue.push(local);
                } else if (queue.top().finalScore() < local.finalScore()) {
                    queue.pop();
                    queue.push(local);
                }
            }
        }
    }

    while (!queue.empty()) {
        auto &e = queue.top();
        if (e.finalScore() != e.finalScore()) {
            printWordIdsWithKeywords(e.getPath(), model_params.lookup_table, word_idf_table);
            cerr << "final score nan" << endl;
            abort();
        }
        if (is_first) {
            int size = e.getPath().size();
            if (size != 1) {
                cerr << boost::format("size is not 1:%1%\n") % size;
                abort();
            }
            searched_ids.insert(e.getPath().at(0).word_id);
        }
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
        printWordIdsWithKeywords(result.getPath(), model_params.lookup_table, word_idf_table);
        ++i;
    }

    return final_results;
}

struct GraphBuilder {
    DynamicLSTMBuilder left_to_right_encoder;

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

            BucketNode *bucket = new BucketNode();
            bucket->init(hyper_params.hidden_dim + hyper_params.word_dim);
            bucket->forward(graph);

            Node *concat = n3ldg_plus::concat(graph, {dropout_node, bucket});

            left_to_right_encoder.forward(graph, model_params.left_to_right_encoder_params,
                    *concat, *hidden_bucket, *hidden_bucket, hyper_params.dropout, is_training);
        }
    }

    void forwardDecoder(Graph &graph, DecoderComponents &decoder_components,
            const std::vector<std::string> &answer,
            const std::vector<std::string> &keywords,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            bool is_training) {
        int keyword_bound = model_params.lookup_table.nVSize;

        for (int i = 0; i < answer.size(); ++i) {
            if (i > 0) {
                keyword_bound = model_params.lookup_table.elems.from_string(keywords.at(i - 1)) + 1;
            }
            int normal_bound = model_params.lookup_table.elems.from_string(keywords.at(i)) + 1;
            if (normal_bound > keyword_bound) {
                print(answer);
                print(keywords);
                abort();
            }
            forwardDecoderByOneStep(graph, decoder_components, i,
                    i == 0 ? nullptr : &answer.at(i - 1), keywords.at(i),
                    i == 0 ||  answer.at(i - 1) == keywords.at(i - 1), hyper_params,
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

        Node *last_input, *last_keyword;
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

            int size = decoder_components.decoder_keyword_lookups.size();
            if (i != size) {
                cerr << boost::format("i is not equal to keyword lookup size i:%1% size:%2%") % i %
                    size << endl;
                abort();
            }
            last_keyword = decoder_components.decoder_keyword_lookups.back();
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
            last_keyword = bucket;
        }

        LookupNode<Param> *keyword_node(new LookupNode<Param>);
        keyword_node->init(hyper_params.word_dim);
        keyword_node->setParam(model_params.lookup_table);
        keyword_node->forward(graph, keyword);
        decoder_components.decoder_keyword_lookups.push_back(keyword_node);

        decoder_components.forward(graph, hyper_params, model_params, *last_input, *last_keyword,
                left_to_right_encoder._hiddens, is_training);

        auto nodes = decoder_components.decoderToWordVectors(graph, hyper_params,
                model_params, left_to_right_encoder._hiddens, i, should_predict_keyword);
        Node *decoder_to_wordvector = nodes.result;
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        LinearWordVectorNode *wordvector_to_onehot(new LinearWordVectorNode);
        wordvector_to_onehot->init(normal_word_id_upper_open_bound);
        wordvector_to_onehot->setParam(model_params.lookup_table.E);
        wordvector_to_onehot->forward(graph, *decoder_to_wordvector);

        Node *softmax = n3ldg_plus::softmax(graph, *wordvector_to_onehot);

        decoder_components.wordvector_to_onehots.push_back(softmax);

        decoder_components.decoder_to_keyword_vectors.push_back(nodes.keyword);

        Node *keyword_vector_to_onehot;
        if (nodes.keyword == nullptr) {
            keyword_vector_to_onehot = nullptr;
        } else {
            keyword_vector_to_onehot = n3ldg_plus::linearWordVector(graph,
                    keyword_word_id_upper_open_bound, model_params.lookup_table.E, *nodes.keyword);
            keyword_vector_to_onehot = n3ldg_plus::softmax(graph, *keyword_vector_to_onehot);
        }
        decoder_components.keyword_vector_to_onehots.push_back(keyword_vector_to_onehot);
    }

    void forwardDecoderResultByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        LookupNode<Param> *keyword_lookup = new LookupNode<Param>;
        keyword_lookup->init(hyper_params.word_dim);
        keyword_lookup->setParam(model_params.lookup_table);
        keyword_lookup->forward(graph, keyword);
        decoder_components.decoder_keyword_lookups.push_back(keyword_lookup);
    }

    void forwardDecoderHiddenByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        Node *last_input, * last_keyword;
        if (i > 0) {
            LookupNode<Param>* before_dropout(new LookupNode<Param>);
            before_dropout->init(hyper_params.word_dim);
            before_dropout->setParam(model_params.lookup_table);
            before_dropout->forward(graph, *answer);
            decoder_components.decoder_lookups_before_dropout.push_back(before_dropout);

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

            if (decoder_components.decoder_keyword_lookups.size() != i) {
                cerr << boost::format("keyword lookup size :%1% i:%2%") %
                    decoder_components.decoder_keyword_lookups.size() % i << endl;
                abort();
            }
            last_keyword = decoder_components.decoder_keyword_lookups.back();
        } else {
            BucketNode *bucket = new BucketNode;
            bucket->init(hyper_params.word_dim);
            bucket->forward(graph);
            last_input = bucket;
            last_keyword = bucket;
        }

        decoder_components.forward(graph, hyper_params, model_params, *last_input, *last_keyword,
                left_to_right_encoder._hiddens, false);
    }

    void forwardDecoderKeywordByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params,
            vector<Node*> &encoder_hiddens) {
        LookupNode<Param> *keyword_embedding = new LookupNode<Param>;
        keyword_embedding->init(hyper_params.word_dim);
        keyword_embedding->setParam(model_params.lookup_table);
        keyword_embedding->forward(graph, keyword);
        if (decoder_components.decoder_keyword_lookups.size() != i) {
            cerr << "keyword lookup size:" << decoder_components.decoder_keyword_lookups.size()
                << endl;
            abort();
        }
        decoder_components.decoder_keyword_lookups.push_back(keyword_embedding);
        ResultAndKeywordVectors result =  decoder_components.decoderToWordVectors(graph,
                hyper_params, model_params, encoder_hiddens, i, false);
        Node *result_node = result.result;

        int keyword_id = model_params.lookup_table.elems.from_string(keyword);
        LinearWordVectorNode *one_hot_node = new LinearWordVectorNode;
        one_hot_node->init(keyword_id + 1);
        one_hot_node->setParam(model_params.lookup_table.E);
        one_hot_node->forward(graph, *result_node);
        Node *softmax = n3ldg_plus::softmax(graph, *one_hot_node);
        decoder_components.wordvector_to_onehots.push_back(softmax);
    }

    pair<vector<WordIdAndProbability>, dtype> forwardDecoderUsingBeamSearch(Graph &graph,
            const vector<DecoderComponents> &decoder_components_beam,
            const unordered_map<string, float> &word_idf_table,
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
//            for (int i = 0; i < closured_count; ++i) {
//                auto & r = word_ids_result.at(i);
//                ++closured_count;
//            }
            if (closured_count >= 1*k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                if (word_ids_result.size() >= 1*k || i > default_config.cut_length) {
                    break;
                }

                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderComponents &decoder_components = beam.at(beam_i);
                    forwardDecoderHiddenByOneStep(graph, decoder_components, i,
                            i == 0 ? nullptr : &last_answers.at(beam_i), hyper_params,
                            model_params);
                }
                cout << "forwardDecoderHiddenByOneStep:" << endl;
                graph.compute();

                most_probable_results = mostProbableKeywords(beam, most_probable_results,
                        word_idf_table, i, k, graph, model_params, hyper_params,
                        default_config, i == 0, searched_ids, black_list);
                for (int beam_i = 0; beam_i < beam.size(); ++beam_i) {
                    DecoderComponents &decoder_components = beam.at(beam_i);
                    int keyword_id = most_probable_results.at(beam_i).getPath().back().word_id;
                    string keyword = model_params.lookup_table.elems.from_id(keyword_id);
                    forwardDecoderResultByOneStep(graph, decoder_components, i, keyword,
                            hyper_params, model_params);
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
                    forwardDecoderKeywordByOneStep(graph, decoder_components, i,
                            last_keywords.at(beam_i), hyper_params, model_params,
                            left_to_right_encoder._hiddens);
                    ++beam_i;
                }
                last_keywords.clear();
                cout << "forwardDecoderKeywordByOneStep:" << endl;
                graph.compute();

                last_answers.clear();
                most_probable_results = mostProbableResults(beam, most_probable_results, i,
                        k, model_params, default_config, i == 0, black_list, word_idf_table);
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

        if (word_ids_result.size() < 1*k) {
            cerr << boost::format("word_ids_result size is %1%, but beam_size is %2%") %
                word_ids_result.size() % k << endl;
            abort();
        }

        cout << endl<< "final search results:" << endl;
        for (const auto &pair : word_ids_result) {
            const vector<WordIdAndProbability> ids = pair.first;
            cout << boost::format("beam result:%1%") % exp(pair.second) << endl;
            printWordIdsWithKeywords(ids, model_params.lookup_table, word_idf_table);
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
