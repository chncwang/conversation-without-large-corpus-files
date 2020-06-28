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

set<std::array<int, 3>> triSet(const vector<int> &sentence) {
    if (sentence.size() < 3) {
        cerr << "triSet" << endl;
        abort();
    }
    using std::array;
    set<array<int, 3>> results;
    for (int i = 0; i < sentence.size() - 2; ++i) {
        array<int, 3> tri = {sentence.at(i + 0), sentence.at(i + 1), sentence.at(i + 2)};
        results.insert(tri);
    }

    if (results.size() != sentence.size() - 2) {
        cerr << boost::format("triSet - result size is %1%, but sentence len is %2%") %
            results.size() % sentence.size() << endl;
        abort();
    }

    return results;
}

set<int> repeatedIds(const vector<int> &sentence) {
    auto tri_set = triSet(sentence);
    set<int> results;
    for (const auto &tri : tri_set) {
        int sentence_len = sentence.size();
        if (tri.at(0) == sentence.at(sentence_len - 2) &&
                tri.at(1) == sentence.at(sentence_len - 1)) {
            results.insert(tri.at(2));
        }
    }
    return results;
}

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

#define BEAM_SEARCH_KEY "beam_search"

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
        set<int> unique_words;
        for (const auto &e : path_) {
            unique_words.insert(e.word_id);
        }
        return final_log_probability / unique_words.size();
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
        const unordered_map<string, float> &idf_table,
        bool print_space = false) {
    cout << "keywords:" << endl;
    for (int i = 0; i < word_ids_with_probability_vector.size(); i += 2) {
        cout << lookup_table.elems.from_id(word_ids_with_probability_vector.at(i).word_id);
        cout << " " << word_ids_with_probability_vector.at(i).word_id << "  ";
    }
    cout << endl;
    cout << "words:" << endl;
    for (int i = 1; i < word_ids_with_probability_vector.size(); i += 2) {
        int word_id = word_ids_with_probability_vector.at(i).word_id;
        cout << lookup_table.elems.from_id(word_id);
        if (print_space && i + 2 < word_ids_with_probability_vector.size()) {
            cout << " ";
        }
    }
    cout << endl;
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
        bool check_tri_gram,
        const vector<string> &black_list,
        const unordered_map<string, float> &idf_table,
        Graph &graph) {
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

    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
        graph.addFLOPs(1, BEAM_SEARCH_KEY);
        return a.finalScore() > b.finalScore();
    };
    priority_queue<BeamSearchResult, vector<BeamSearchResult>, decltype(cmp)> queue(cmp);
    vector<BeamSearchResult> results;
//    for (int i = 0; i < (is_first ? 1 : nodes.size()); ++i) {
    for (int i = 0; i < nodes.size(); ++i) {
        const Node &node = *nodes.at(i);

        BeamSearchResult beam_search_result;
        set<int> repeated_ids;
        if (check_tri_gram) {
            vector<int> word_ids;
            int m = 0;
            for (const auto &e : last_results.at(i).getPath()) {
                if (m++ % 2 == 0) {
                    continue;
                }
                word_ids.push_back(e.word_id);
            }
            repeated_ids = repeatedIds(word_ids);
        }
        for (int j = 0; j < nodes.at(i)->getDim(); ++j) {
            if (repeated_ids.find(j) != repeated_ids.end()) {
                continue;
            }
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
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            dtype word_probability = value;
            vector<WordIdAndProbability> word_ids;
            if (!last_results.empty()) {
                log_probability += last_results.at(i).finalLogProbability();
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                word_ids = last_results.at(i).getPath();
            }
            word_ids.push_back(WordIdAndProbability(node.getDim(), j, word_probability));
            beam_search_result =  BeamSearchResult(beam.at(i), word_ids, log_probability);
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            int local_size = k;
            if (queue.size() < local_size) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                queue.pop();
                queue.push(beam_search_result);
            } else {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
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
        bool should_predict_keyword = is_first;
        Node *node, *keyword_node, *hidden;
        hidden = beam.at(ii).decoder._hiddens.at(word_pos);
        if (should_predict_keyword) {
            DecoderComponents &components = beam.at(ii);

            if (components.decoder_lookups.size() != word_pos) {
                cerr << boost::format("size:%1% word_pos:%2%") % components.decoder_lookups.size()
                    % word_pos << endl;
                abort();
            }

            Node *context_concated = n3ldg_plus::concat(graph,
                    {components.decoder._hiddens.at(word_pos), components.contexts.at(word_pos)});

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

            Node *keyword_vector_to_onehot = n3ldg_plus::linearWordVector(graph,
                    last_keyword_id + 1, model_params.lookup_table.E, *keyword);
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

    auto cmp = [&](const BeamSearchResult &a, const BeamSearchResult &b) {
        graph.addFLOPs(1, BEAM_SEARCH_KEY);
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
            graph.addFLOPs(1, BEAM_SEARCH_KEY);
            if (queue.size() < k) {
                queue.push(beam_search_result);
            } else if (queue.top().finalScore() < beam_search_result.finalScore()) {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                queue.pop();
                queue.push(beam_search_result);
            } else {
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
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

                if (word_idf_table.at(model_params.lookup_table.elems.from_id(j)) >= 9.0f) {
                    break;
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
                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
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
                graph.addFLOPs(1, BEAM_SEARCH_KEY);
                if (queue.size() < k) {
                    queue.push(local);
                } else if (queue.top().finalScore() < local.finalScore()) {
                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
                    queue.pop();
                    queue.push(local);
                } else {
                    graph.addFLOPs(1, BEAM_SEARCH_KEY);
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
        Node *hidden_bucket = n3ldg_plus::bucket(graph, hyper_params.hidden_dim, 0);
        for (int i = 0; i < sentence.size(); ++i) {
            Node *input_lookup = n3ldg_plus::embedding(graph, model_params.lookup_table,
                    sentence.at(i));
            Node *dropout_node = n3ldg_plus::dropout(graph, *input_lookup, hyper_params.dropout,
                    is_training);
            left_to_right_encoder.forward(graph, model_params.left_to_right_encoder_params,
                    *dropout_node, *hidden_bucket, *hidden_bucket, hyper_params.dropout,
                    is_training);
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
            int normal_bound = model_params.lookup_table.elems.from_string(keywords.at(0)) + 1;
            if (normal_bound > keyword_bound) {
                print(answer);
                print(keywords);
                abort();
            }

            const string &keyword = i == 0 ? keywords.front() : STOP_SYMBOL;
            forwardDecoderByOneStep(graph, decoder_components, i, i == answer.size() - 1,
                    i == 0 ? nullptr : &answer.at(i - 1), keyword, hyper_params,
                    model_params, is_training, keyword_bound, normal_bound);
        }
    }

    void forwardDecoderByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            bool is_last,
            const std::string *answer,
            const std::string &keyword,
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
            Node *before_dropout = n3ldg_plus::embedding(graph, model_params.lookup_table,
                    *answer);
            Node *decoder_lookup = n3ldg_plus::dropout(graph, *before_dropout,
                    hyper_params.dropout, is_training);
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
            Node *bucket = n3ldg_plus::bucket(graph, hyper_params.word_dim, 0);
            last_input = bucket;
            last_keyword = bucket;
        }

        Node *keyword_node = n3ldg_plus::embedding(graph, model_params.lookup_table, keyword);
        Node * dropout_keyword = n3ldg_plus::dropout(graph, *keyword_node, hyper_params.dropout,
                is_training);
        decoder_components.decoder_keyword_lookups.push_back(dropout_keyword);

        decoder_components.forward(graph, hyper_params, model_params, *last_input, *last_keyword,
                left_to_right_encoder._hiddens, is_training);

        auto nodes = decoder_components.decoderToWordVectors(graph, hyper_params,
                model_params, left_to_right_encoder._hiddens, i);
        Node *decoder_to_wordvector = nodes.result;
        decoder_components.decoder_to_wordvectors.push_back(decoder_to_wordvector);

        Node *wordvector_to_onehot = n3ldg_plus::linearWordVector(graph,
                normal_word_id_upper_open_bound, model_params.lookup_table.E,
                *decoder_to_wordvector);

        Node *softmax = n3ldg_plus::softmax(graph, *wordvector_to_onehot);

        decoder_components.wordvector_to_onehots.push_back(softmax);

        decoder_components.decoder_to_keyword_vectors.push_back(nodes.keyword);

        Node *keyword_vector_to_onehot;
        if (nodes.keyword == nullptr) {
            keyword_vector_to_onehot = nullptr;
        } else {
            keyword_vector_to_onehot = n3ldg_plus::linearWordVector(graph,
                    keyword_word_id_upper_open_bound, model_params.lookup_table.E,
                    *nodes.keyword);
            keyword_vector_to_onehot = n3ldg_plus::softmax(graph, *keyword_vector_to_onehot);
        }
        decoder_components.keyword_vector_to_onehots.push_back(keyword_vector_to_onehot);
    }

    void forwardDecoderResultByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const string &keyword,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        Node *keyword_lookup = n3ldg_plus::embedding(graph, model_params.lookup_table, keyword);
        Node *dropout_keyword = n3ldg_plus::dropout(graph, *keyword_lookup, hyper_params.dropout,
                false);
        decoder_components.decoder_keyword_lookups.push_back(dropout_keyword);
    }

    void forwardDecoderHiddenByOneStep(Graph &graph, DecoderComponents &decoder_components, int i,
            const std::string *answer,
            const HyperParams &hyper_params,
            ModelParams &model_params) {
        Node *last_input, * last_keyword;
        if (i > 0) {
            Node *before_dropout = n3ldg_plus::embedding(graph, model_params.lookup_table,
                    *answer);

            Node *decoder_lookup = n3ldg_plus::dropout(graph, *before_dropout,
                    hyper_params.dropout, false);
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
            Node *bucket = n3ldg_plus::bucket(graph, hyper_params.word_dim, 0);
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
        Node *keyword_embedding = n3ldg_plus::embedding(graph, model_params.lookup_table,
                keyword);
        if (decoder_components.decoder_keyword_lookups.size() != i) {
            cerr << "keyword lookup size:" << decoder_components.decoder_keyword_lookups.size()
                << endl;
            abort();
        }
        decoder_components.decoder_keyword_lookups.push_back(keyword_embedding);
        ResultAndKeywordVectors result =  decoder_components.decoderToWordVectors(graph,
                hyper_params, model_params, encoder_hiddens, i);
        Node *result_node = result.result;

        int keyword_id = model_params.lookup_table.elems.from_string(keyword);
        Node *one_hot_node = n3ldg_plus::linearWordVector(graph, keyword_id + 1,
                model_params.lookup_table.E, *result_node);
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
            if (closured_count >= default_config.result_count_factor * k) {
                break;
            }

            for (int i = 0;; ++i) {
                cout << boost::format("forwardDecoderUsingBeamSearch i:%1%\n") % i;
                if (word_ids_result.size() >= default_config.result_count_factor * k ||
                        i > default_config.cut_length) {
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
                cout << "forwardDecoderKeywordByOneStep: graph computed" << endl;

                last_answers.clear();
                most_probable_results = mostProbableResults(beam, most_probable_results, i,
                        k, model_params, default_config, i == 0, i>=3, black_list, word_idf_table,
                        graph);
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
